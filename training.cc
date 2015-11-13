#include "training.hpp"

#include <boost/format.hpp>

// For logging
#define ELPP_NO_DEFAULT_LOG_FILE
#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP


// ********************************************************
// train
// ********************************************************
int train(char* ftrn, char* fdev, unsigned nlayers, 
	  unsigned inputdim, unsigned hiddendim, 
	  string flag, float lr0, bool use_adagrad, string fmodel){
  // initialize logging
  int argc = 1; 
  char** argv = new char* [1];
  START_EASYLOGGINGPP(argc, argv);
  delete[] argv;
  
  // ---------------------------------------------
  // 
  int kSOS, kEOS;
  cnn::Dict d;
  string MODELPATH("models/");
  string LOGPATH("logs/");
  
  // ---------------------------------------------
  // predefined files
  ostringstream os;
  os << flag << '_' << nlayers << '_' << inputdim
     << '_' << hiddendim << '_' << lr0 << '_' << use_adagrad
     << "-pid" << getpid();
  const string fprefix = os.str();
  string fname = MODELPATH + fprefix;
  string flog = LOGPATH + fprefix + ".log";
  // check model path
  check_dir(MODELPATH);

  // ----------------------------------------------
  // Pre-defined constants
  double best = 9e+99;
  unsigned report_every_i = 50; // 50
  unsigned dev_every_i_reports = 20; // 20


  // --------------------------------------------
  // Logging
  el::Configurations defaultConf;
  // defaultConf.setToDefault();
  defaultConf.set(el::Level::Info, 
  		  el::ConfigurationType::Format, 
  		  "%datetime{%h:%m:%s} %level %msg");
  defaultConf.set(el::Level::Info, 
  		  el::ConfigurationType::Filename, flog.c_str());
  el::Loggers::reconfigureLogger("default", defaultConf);
  LOG(INFO) << "Training data: " << ftrn;
  LOG(INFO) << "Dev data: " << fdev;
  LOG(INFO) << "Parameters will be written to: " << fname;

  // ---------------------------------------------
  // Either create a dict or load one from the model file
  Corpus training, dev;
  if (fmodel.size() == 0){
    kSOS = d.Convert("<s>"); //Convert is a method of dict
    kEOS = d.Convert("</s>");
    LOG(INFO) << "Create dict from training data ...";
    // read training data
    training = readData(ftrn, &d, true);
    // no new word types allowed
    d.Freeze(); 
    // reading dev data
    dev = readData(fdev, &d, false);
  } else {
    LOG(INFO) << "Load dict from: " << fmodel;
    ifstream in(fmodel + ".dict");
    boost::archive::text_iarchive ia(in);
    ia >> d; d.Freeze(); 
    training = readData(ftrn, &d, false);
    dev = readData(fdev, &d, false);
  }
  // get dict size
  unsigned vocabsize = d.size();
  LOG(INFO) << "Vocab size = " << vocabsize;
  // save dict
  save_dict(fname, d);
  LOG(INFO) << "Save dict into: " << fname;
  // segment training doc
  int len_thresh = 5;
  LOG(INFO) << "Length threshold: " << len_thresh;
  training = segment_doc(training, len_thresh);
  LOG(INFO) << "New training set size: " << training.size();

  // ----------------------------------------------
  // define model
  Model omodel, hmodel, rmodel, smodel, wmodel;
  // only one of them is used in the following
  DCLMOutput<LSTMBuilder> olm(omodel, nlayers, inputdim, 
			      hiddendim, vocabsize);
  DCLMHidden<LSTMBuilder> hlm(hmodel, nlayers, inputdim, 
			      hiddendim, vocabsize);
  RNNLM<LSTMBuilder> rnnlm(rmodel, nlayers, inputdim,
			   hiddendim, vocabsize);
  HRNNLM<LSTMBuilder> hrnnlm(smodel, wmodel, nlayers, 
			     inputdim,
			     hiddendim, vocabsize);
  // Load model
  if (fmodel.size() > 0){
    LOG(INFO) << "Load model from: " << fmodel;
    if (flag == "rnnlm"){
      load_model(fprefix, rmodel);
    } else if (flag == "output"){
      load_model(fprefix, omodel);
    } else if (flag == "hidden"){
      load_model(fprefix, hmodel);
    } else if (flag == "hrnnlm"){
      LOG(INFO) << "Cannot handle this case now ...";
      return -1;
    }
  } else {
    LOG(INFO) << "Randomly initializing model parameters ...";
  }
  // define learner
  Trainer* sgd = nullptr;
  Trainer* sgd2 = nullptr; // only for hrnnlm
  if (flag == "rnnlm"){
    sgd = new SimpleSGDTrainer(&rmodel, 1e-6, lr0); 
  } else if (flag == "output") {
    sgd = new SimpleSGDTrainer(&omodel, 1e-6, lr0);     
  } else if (flag == "hidden") {
    sgd = new SimpleSGDTrainer(&hmodel, 1e-6, lr0);     
  } else if (flag == "hrnnlm") { 
    sgd = new SimpleSGDTrainer(&smodel, 1e-6, lr0);
    sgd2 = new SimpleSGDTrainer(&wmodel, 1e-6, lr0);    
  } else {
    LOG(INFO) << "Unrecognized flag";
    return -1;    
  }
    
  // ---------------------------------------------
  // define the indices so we can shuffle the docs
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  bool first = true; int report = 0; unsigned lines = 0;

  // ---------------------------------------------
  // start training
  unsigned si = training.size();
  while(true) {
    // Timer iteration("completed in");
    double dloss = 0, loss = 0;
    unsigned words = 0, dwords = 0;
    //iterating over documents
    for (unsigned i = 0; i < report_every_i; ++i) { 
      //check if it's the number of documents
      if (si == training.size()) { 
        si = 0;
        if (first) { 
	  first = false; 
	} else { 
	  sgd->update_epoch(); 
	  if (flag == "hrnnlm")
	    sgd2->update_epoch();
	}
	cout << "==SHUFFLE==" << endl;
        shuffle(order.begin(), order.end(), *rndeng);
      }
      // get one document
      auto& doc = training[order[si]];
      // get how many words in this documents
      dwords = 0;
      for (auto& sent : doc) 
	dwords += (sent.size() - 1);
      ComputationGraph cg;
      // get the right model
      if (flag == "rnnlm"){
	rnnlm.BuildGraph(doc, cg);
      } else if (flag == "output") {
	olm.BuildGraph(doc, cg);	
      } else if (flag == "hidden") {
	hlm.BuildGraph(doc, cg);	
      } else if (flag == "hrnnlm") {
	// stensor.clear();
	if (doc.size() > 1){
	  // ignore, if only has one sentence
	  hrnnlm.BuildSentGraph(doc, cg); 
	}
      }
      // run forward and backward for sgd update
      if ((flag != "hrnnlm") || (doc.size() > 1)){
	// ignore, if it is hrnnlm and doc only 
	//   has one sentence
	dloss = as_scalar(cg.forward());
	cg.backward(); 
	sgd->update();
      } 
      // else {
      // 	cout << "flag = " << flag 
      // 	     << " doc.size = " << doc.size() << endl;
      // }
      if (flag == "hrnnlm"){
      	// update word-level LM for hrnnlm
      	hrnnlm.BuildWordGraph(doc, cg);
	// update dloss
	dloss = as_scalar(cg.forward());
	// backward and update
      	cg.backward(); 
	sgd2->update();
      } 
      loss += dloss; words += dwords;
      si ++;
    }
    sgd->status();
    if (flag == "hrnnlm") { sgd2->status(); }
    LOG(INFO) << " E = " 
	      << boost::format("%1.4f") % (loss / words) 
	      << " PPL = " 
	      << boost::format("%5.4f") % exp(loss / words) 
	      << ' ';
    
    // ----------------------------------------
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      int dwords = 0, docctr = 0;
      for (auto& doc : dev){
	// for each doc
	ComputationGraph cg;
	// get the right model
	if (flag == "output") {
	  olm.BuildGraph(doc, cg);
	} else if (flag == "hidden") {
	  hlm.BuildGraph(doc, cg);
	} else if (flag == "rnnlm") {
	  rnnlm.BuildGraph(doc, cg);
	} else if (flag == "hrnnlm") {
	  // for compute stensor
	  if (doc.size() > 1){
	    hrnnlm.BuildSentGraph(doc, cg);
	    cg.forward();
	  }
	  // build graph on word-level LM
	  hrnnlm.BuildWordGraph(doc, cg);
	  // get word-level loss from the 
	  //   next forward
	}
	dloss += as_scalar(cg.forward());
	for (auto& sent : doc) dwords += sent.size() - 1;
      }
      // print PPL on dev
      LOG(INFO) << "DEV[epoch=" 
		<< (lines / (double)training.size()) 
		<< "] E = "
		<< boost::format("%1.4f") % (dloss / dwords) 
		<< " PPL = " 
		<< boost::format("%5.4f") % exp(dloss / dwords) 
		<< " ("
		<< boost::format("%5.4f") % exp(best / dwords)
		<<") ";
      // Save model
      if (dloss < best) {
	best = dloss;
	LOG(INFO) << "Save model into: "<<fname;
	if (flag == "rnnlm"){
	  save_model(fname, rmodel);
	} else if (flag == "output"){
	  save_model(fname, omodel);
	} else if (flag == "hidden"){
	  save_model(fname, hmodel);
	} else if (flag == "hrnnlm"){
	  save_model(fname + ".sent", smodel);
	  save_model(fname + ".word", wmodel);
	}
      }
    }
    // end dev
  }
  delete sgd, sgd2;
}
