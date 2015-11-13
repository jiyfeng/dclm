#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include "util.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>

using namespace std;
using namespace cnn;

namespace po = boost::program_options;

// For logging
#define ELPP_NO_DEFAULT_LOG_FILE
#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

unsigned LAYERS = 2;
unsigned INPUT_DIM = 16;  //256
unsigned HIDDEN_DIM = 48;  // 1024
unsigned VOCAB_SIZE = 0;
char DOC_DELIM = '=';
string SENT_DELIM = "<s>";
unsigned REPORT_EVERY_I = 50;
string FPREFIX;

cnn::Dict d;
int kSOS, kEOS;
string MODELPATH("models/");
string LOGPATH("logs/");


template <class Builder>
struct RNNLanguageModel {
  LookupParameters* p_c; //word embeddings VxK1
  Parameters* p_R; //recurrence weights VxK2
  Parameters* p_bias; //bias Vx1
  Builder builder;
  
  explicit RNNLanguageModel(Model& model) : builder(LAYERS, INPUT_DIM, 
						    HIDDEN_DIM, &model) {
    p_c = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM}); 
    p_R = model.add_parameters({VOCAB_SIZE, HIDDEN_DIM});
    p_bias = model.add_parameters({VOCAB_SIZE});
  }

  // return Expression of total loss
  Expression BuildLMGraph(const vector<int>& sent, ComputationGraph& cg) {
    const unsigned slen = sent.size() - 1;
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
    Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
    Expression i_bias = parameter(cg, p_bias);  // word bias

    vector<Expression> errs;
    for (unsigned t = 0; t < slen; ++t) {
      Expression i_x_t = lookup(cg, p_c, sent[t]); 
      Expression i_y_t = builder.add_input(i_x_t); 
      Expression i_r_t =  i_bias + i_R * i_y_t;

      Expression i_err = pickneglogsoftmax(i_r_t, sent[t+1]);
      errs.push_back(i_err);
    }
    Expression i_nerr = sum(errs);
    return i_nerr;
  }
};


int train(string ftrn, string fdev){
  // -------------------------------------------
  LOG(INFO) << "Training data: " << ftrn;
  Corpus training = readData((char*) ftrn.c_str(), &d, true);
  d.Freeze(); VOCAB_SIZE = d.size();
  LOG(INFO) << "Dev data: " << fdev;
  Corpus dev = readData((char*) fdev.c_str(), &d, false);

  // -------------------------------------------
  string fname = MODELPATH + FPREFIX;
  LOG(INFO) << "Save dict into: " << fname << ".dict";
  LOG(INFO) << "Parameters will be written to: " << fname << ".model";
  save_dict(fname, d);
  // check model path
  check_dir(MODELPATH);
  double best = 9e+99;

  // -------------------------------------------
  Model model;
  Trainer* sgd = nullptr;
  sgd = new SimpleSGDTrainer(&model);
  RNNLanguageModel<LSTMBuilder> lm(model);

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
    unsigned lines = 0;
    unsigned words = 0;
    for (unsigned i = 0; i < REPORT_EVERY_I; ++i) {
      if (si == training.size()) {
        si = 0;
        if (first) { first = false; } 
	else { sgd->update_epoch(); }
        shuffle(order.begin(), order.end(), *rndeng);
      }
      auto& doc = training[order[si]];

      for (unsigned j = 0; j < doc.size(); j++){
	// build graph for this instance
	ComputationGraph cg;
	auto& sent = doc[j];
	words += sent.size();
	lm.BuildLMGraph(sent,cg);
	loss += as_scalar(cg.forward());
	cg.backward();
	sgd->update();
	++lines;
      }
      ++si;
    }
    sgd->status();
    LOG(INFO) << " E = " 
	      << boost::format("%1.4f") % (loss / words) 
	      << " PPL = " 
	      << boost::format("%5.4f") % exp(loss / words);

    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      int dwords = 0;
      int docctr = 0;
      for (auto& doc : dev){
	for (auto& sent : doc){
	  ComputationGraph cg;
	  lm.BuildLMGraph(sent, cg);
	  dloss += as_scalar(cg.forward());
	  dwords += sent.size() - 1;
	}
      }
      LOG(INFO) << "DEV [epoch=" 
		<< (lines / (double)training.size()) 
		<< "] E = " 
		<< boost::format("%1.4f") % (dloss / dwords) 
		<< " PPL = " 
		<< boost::format("%5.4f") % exp(dloss / dwords)
		<< " ("
		<< boost::format("%5.4f") % exp(best / dwords)
		<<") ";
      if (dloss < best) {
        best = dloss;
	LOG(INFO) << "Save model into: "<<fname;
	save_model(fname, model);
      }
    }
  }
  delete sgd;
  return 0;
}

int test(string fprefix, string ftst){
  // -------------------------------------------
  cerr << "Load dict from: " << fprefix << ".dict" << endl;
  load_dict(fprefix, d);
  d.Freeze(); VOCAB_SIZE = d.size();
  cerr << "Test data: " << ftst;
  // call the readData from util.hpp
  Corpus tst = readData((char*) ftst.c_str(), &d, false);

  // -------------------------------------------
  Model model;
  RNNLanguageModel<LSTMBuilder> lm(model);
  cerr << "Load model from: " << fprefix << endl;
  load_model(fprefix, model);

  double loss = 0, dloss = 0;
  int dwords = 0, words = 0;
  for (auto& doc : tst){
    for (auto& sent : doc){
      ComputationGraph cg;
      lm.BuildLMGraph(sent, cg);
      dwords = sent.size() - 1;
      dloss = as_scalar(cg.forward());
      words += dwords;
      loss += dloss;
    }
  }
  cerr << "PPL = "
       << boost::format("%5.4f") % exp(loss / words) << endl;
  return 0;
}

/** goal: modify this to read ptb data instead, respecting doc boundaries **/
int main(int argc, char** argv) {
  cnn::Initialize(argc, argv); 
  START_EASYLOGGINGPP(argc, argv);
  // --------------------------------------------
  // Paragram options
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("action", po::value<string>(), "training or test")
    ("training-file", po::value<string>(), "training file")
    ("dev-file", po::value<string>(), "dev file")
    ("test-file", po::value<string>(), "test file")
    ("model-file", po::value<string>(), "model file")
    ("layers", po::value<int>()->default_value((int)2), "number of RNN layers")
    ("input-dim", po::value<int>()->default_value((int)16), "input dimension")
    ("hidden-dim", po::value<int>()->default_value((int)48), "hidden dimension")
    ("report-stride", po::value<int>()->default_value((int)50), "report every i iterations");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    
  // 
  po::positional_options_description p;
  p.add("training-file",1);
  p.add("dev-file",2);
  // 
  po::store(po::command_line_parser(argc,argv).
	    options(desc).positional(p).run(), vm);
	    po::notify(vm);
  if (vm.count("help")) { cout << desc << "\n"; return 1;}

  // ------------------------------------------------
  string ACTION = vm["action"].as<string>();
  LAYERS = vm["layers"].as<int>();
  INPUT_DIM = vm["input-dim"].as<int>();
  HIDDEN_DIM = vm["hidden-dim"].as<int>();
  REPORT_EVERY_I = vm["report-stride"].as<int>();
  cerr << LAYERS << " " << INPUT_DIM << " " 
       << HIDDEN_DIM << " " << REPORT_EVERY_I;
  // -------------------------------------------------
  // Take different actions
  if (ACTION == "train"){
    if ((!vm.count("training-file"))||(!vm.count("dev-file"))){
      cerr << "Must specify a training and a dev file" << endl;
      return -1;
    }
    // ------------------------------------------------
    ostringstream os;
    os << "baseline" << '_' << LAYERS << '_' << INPUT_DIM 
       << '_' << HIDDEN_DIM << "-pid" << getpid();
    FPREFIX = os.str();
    string flog = LOGPATH + FPREFIX + ".log";
    // ------------------------------------------------
    // Logging
    
    el::Configurations defaultConf;
    // defaultConf.setToDefault();
    defaultConf.set(el::Level::Info, 
		    el::ConfigurationType::Format, 
		    "%datetime{%h:%m:%s} %level %msg");
    defaultConf.set(el::Level::Info, 
		    el::ConfigurationType::Filename, flog.c_str());
    el::Loggers::reconfigureLogger("default", defaultConf);
    // -----------------------------------------------
    string ftrn = vm["training-file"].as<string>();
    string fdev = vm["dev-file"].as<string>();
    train(ftrn, fdev);
  } 
  else if (ACTION == "test"){
    if ((!vm.count("model-file"))||(!vm.count("test-file"))){
      cerr << "Must specify a model and a test file" << endl;
      return -1;
    }
    string ftst = vm["test-file"].as<string>();
    string fprefix = vm["model-file"].as<string>();
    test(fprefix, ftst);
  }
}
