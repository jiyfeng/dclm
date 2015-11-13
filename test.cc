#include "test.hpp"

#include <boost/format.hpp>

// ********************************************************
// test
// ********************************************************
int test(char* ftst, char* prefix, string flag){
  // ---------------------------------------------
  // 
  cnn::Dict d;

  // ---------------------------------------------
  // predefined variable (will be overwritten after 
  //    loading model)
  unsigned nlayers = 2;
  unsigned inputdim = 16, hiddendim = 48;
  if (flag.size() == 0) flag = "output";
  // model and dict file name prefix
  string fprefix = string(prefix);
  string fout = string(ftst);
  fout += ("." + flag + ".result");
  ofstream myfile; myfile.open(fout);

  // ---------------------------------------------
  // check model name
  if (fprefix.size() == 0){
    cerr << "Unspecified model name" << endl;
    return -1;
  }
  // load dict and freeze it
  load_dict(fprefix, d);
  unsigned vocabsize = d.size();
  cerr << "Vocab size = " << vocabsize << endl;
  d.Freeze();
  Corpus tst = readData(ftst, &d, false);

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
			     inputdim, hiddendim, vocabsize);
  
  // Load model
  cerr << "Load model from: " << fprefix << ".model" << endl;
  if (flag == "rnnlm"){
    load_model(fprefix, rmodel);
  } else if (flag == "output"){
    load_model(fprefix, omodel);
  } else if (flag == "hidden"){
    load_model(fprefix, hmodel);
  } else if (flag == "hrnnlm"){
    load_model(fprefix + ".sent", smodel);
    load_model(fprefix + ".word", wmodel);
  } else {
    cerr << "Unrecognized flag" << endl;
    return -1;
  }

  // ---------------------------------------------
  // start testing
  double loss = 0, dloss = 0;
  unsigned words = 0, dwords = 0;
  //iterating over documents
  for (auto& doc : tst){
    ComputationGraph cg;
    // get the right model
    if (flag == "output"){
      olm.BuildGraph(doc, cg);
    } else if (flag == "hidden"){
      hlm.BuildGraph(doc, cg);
    } else if (flag == "rnnlm"){
      rnnlm.BuildGraph(doc, cg);
    } else if (flag == "hrnnlm"){
      hrnnlm.BuildSentGraph(doc, cg);
    } else {
      cerr << "unrecognized flag " << endl;
      return -1;
    }
    // run forward and backward for sgd update
    dloss = as_scalar(cg.forward());
    dwords = 0;
    for (auto& sent : doc) dwords += (sent.size() - 1);
    loss += dloss;
    words += dwords;
    cerr << boost::format("%5.4f") % exp(dloss / dwords)
	 << endl;
    myfile << " PPL = " 
	   << boost::format("%5.4f") % exp(dloss / dwords)
	   << endl;
  }
  cerr << " E = " 
       << boost::format("%1.4f") % (loss / words) 
       << " PPL = " 
       << boost::format("%5.4f") % exp(loss / words) 
       << endl;
  myfile.close();
}
