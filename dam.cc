#include "dam.h"

#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"

#include "util.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>
#include <set>
#include <map>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/format.hpp>

using namespace std;
using namespace cnn;

// For logging
#define ELPP_NO_DEFAULT_LOG_FILE
#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

unsigned LAYERS = 2;      // 2
unsigned INPUTDIM = 96;
unsigned HIDDENDIM = 96;
unsigned ALIGNDIM = 48;
unsigned VOCAB_SIZE = 0;

cnn::Dict d;
int kEOS, kSOS;

string MODELPATH("models/");
string LOGPATH("logs/");

typedef Sent Sentence;
typedef Doc Document;
// typedef vector<Document> Corpus;

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;

void read_documents(char* fname, Corpus &corpus, bool b_update) {
  corpus = readData(fname, &d, b_update);
}

int train(char* ftrn, char* fdev, string fname){
  // ---------------------------------------------
  // setup
  check_dir(MODELPATH);
  double best = 9e+99;
  
  // --------------------------------------------
  // load the corpora
  Corpus training, dev;
  LOG(INFO) << "Reading training data from: " << ftrn;
  read_documents(ftrn, training, true);
  int len_thresh = 5;
  LOG(INFO) << "Length threshold: " << len_thresh;
  training = segment_doc(training, len_thresh);
  LOG(INFO) << "New training set size: " << training.size();
  d.Freeze(); VOCAB_SIZE = d.size();
  LOG(INFO) << "Parameters will be written to: " << fname;
  LOG(INFO) << "Save dict into: " << fname;
  save_dict(fname, d);
  LOG(INFO) << "Reading dev data from: " << fdev;
  read_documents(fdev, dev, false);

  // --------------------------------------------
  // define model
  Model model;
  Trainer* sgd = new SimpleSGDTrainer(&model);
  DocumentAttentionalModel<LSTMBuilder> lm(model, VOCAB_SIZE, 
					   LAYERS, INPUTDIM, 
					   HIDDENDIM, ALIGNDIM);
  
  // --------------------------------------------
  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 20;
  unsigned si = 0;
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  bool first = true;
  int report = 0;
  unsigned lines = 0;
  while(true) {
    Timer iteration("completed in");
    double loss = 0;
    unsigned chars = 0;
    for (unsigned i = 0; i < report_every_i; ++i) {
      if (si == training.size()) {
	si = 0;
	if (first) { first = false; } 
	else { sgd->update_epoch(); }
	LOG(INFO) << "*** SHUFFLE ***" << endl;
	shuffle(order.begin(), order.end(), *rndeng);
      }
      
      // build graph for this instance
      ComputationGraph cg;
      auto& doc = training[order[si]];
      for (auto &sent: doc)
	chars += sent.size() - 1;
      ++si;
      //cerr << "sent length " << sent.size();
      lm.BuildGraph(doc, cg);
      loss += as_scalar(cg.forward());
      cg.backward();
      sgd->update();
      ++lines;
    }
    sgd->status();
    // FIXME: is chars incorrect?
    LOG(INFO) << " E = " 
	      << boost::format("%1.4f") % (loss / chars) 
	      << " PPL = " 
	      << boost::format("%5.4f") % exp(loss / chars);
    
    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      int dchars = 0;
      for (unsigned i = 0; i < dev.size(); ++i) {
	const auto& doc = dev[i];
	ComputationGraph cg;
	lm.BuildGraph(doc, cg);
	dloss += as_scalar(cg.forward());
	for (auto &sent: doc)
	  dchars += sent.size() - 1;
      }
      LOG(INFO) << "DEV [epoch = " 
		<< (lines / (double)training.size()) 
		<< "] E = " 
		<< boost::format("%1.4f") % (dloss / dchars) 
		<< " PPL = " 
		<< boost::format("%5.4f") % exp(dloss / dchars)
		<< " ("
		<< boost::format("%5.4f") % exp(best / dchars)
		<< ") ";
      if (dloss < best) {
	best = dloss;
	LOG(INFO) << "Save model into: " << fname;
	save_model(fname, model);
      }
    }
  }
  delete sgd;
  return 0;
}

int test(char* ftst, string fmodel){
  // -------------------------------------------
  // load dict
  cerr << "Load dict from: " << fmodel << endl;
  load_dict(fmodel, d);
  d.Freeze(); VOCAB_SIZE = d.size();
  cerr << "Vocab size = " << VOCAB_SIZE << endl;
  // -------------------------------------------
  // load data
  cerr << "Read data from: " << ftst << endl;
  Corpus tst;
  read_documents(ftst, tst, false);

  // --------------------------------------------
  // define model
  Model model;
  DocumentAttentionalModel<LSTMBuilder> lm(model, VOCAB_SIZE, 
					   LAYERS, INPUTDIM, 
					   HIDDENDIM, ALIGNDIM);
  // --------------------------------------------
  // load model
  cerr << "Load model from: " << fmodel << endl;
  load_model(fmodel, model);

  // --------------------------------------------
  // run test
  string fout = string(ftst);
  fout += ".dam.result";
  ofstream myfile; myfile.open(fout);
  double loss = 0, dloss = 0;
  int words = 0, dwords = 0;
  cerr << "Start computing ..." << endl;
  for (auto& doc : tst){
    ComputationGraph cg;
    lm.BuildGraph(doc, cg);
    dloss = as_scalar(cg.forward());
    loss += dloss;
    dwords = 0;
    for (auto& sent : doc) dwords += (sent.size() - 1);
    words += dwords;
    cerr << boost::format("%5.4f") % exp(dloss / dwords)
	 << endl;
    myfile << boost::format("%5.4f") % exp(dloss / dwords)
	 << endl;
  }
  cerr << " E = " 
       << boost::format("%1.4f") % (loss / words) 
       << " PPL = " 
       << boost::format("%5.4f") % exp(loss / words) 
       << endl;
  myfile.close();
  return 0;
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  
  // check arguments
  cout<<"Number of arguments "<<argc<<endl;
  if (argc < 4) {
    cerr << "Usage: \n" 
	 <<"\t" << argv[0] 
	 << " train train_file dev_file [input_dim] [hidden_dim] [align_dim]\n"
	 <<"\t" << argv[0] 
	 << " test model_prefix test_file\n";
    return 1;
  }

  // --------------------------------------------
  // parse command arguments
  string cmd = argv[1];
  cout << "Task: " << cmd <<endl;
  if (cmd == "train"){
    char* ftrn = argv[2];
    char* fdev = argv[3];
    // -------------------------------------------
    if (argc >= 5) INPUTDIM = atoi(argv[4]);
    if (argc >= 6) HIDDENDIM = atoi(argv[5]);
    if (argc >= 7) ALIGNDIM = atoi(argv[6]);
    // --------------------------------------------
    ostringstream os;
    os << "dam" << '_' << LAYERS << '_' << INPUTDIM
       << '_' << HIDDENDIM << '_' << ALIGNDIM
       << "-pid" << getpid();
    string fprefix = os.str();
    // --------------------------------------------
    // Logging
    string flog = LOGPATH + fprefix + ".log";
    el::Configurations defaultConf;
    // defaultConf.setToDefault();
    defaultConf.set(el::Level::Info, 
		    el::ConfigurationType::Format, 
		  "%datetime{%h:%m:%s} %level %msg");
    defaultConf.set(el::Level::Info, 
		    el::ConfigurationType::Filename, 
		    flog.c_str());
    el::Loggers::reconfigureLogger("default", defaultConf);
    // --------------------------------------------
    string fname = MODELPATH + fprefix;
    LOG(INFO) << "Training data: " << ftrn;
    LOG(INFO) << "Dev data: " << fdev;
    train(ftrn, fdev, fname);
  } else if (cmd == "test"){
    char* ftst = argv[3];
    string fmodel = argv[2];
    test(ftst, fmodel);
    return -1;
  }
  
  return 0;
}

