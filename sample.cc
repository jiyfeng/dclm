#include "sample.hpp"


// ********************************************************
// test
// ********************************************************
int randomsample(char* fcontext, char* prefix, string flag){
  cnn::Dict d;
  // ---------------------------------------------
  // predefined variable (will be overwritten after 
  //    loading model)
  unsigned nlayers = 2;
  unsigned inputdim = 16, hiddendim = 48;
  if (flag.size() == 0){
    cerr << "Unspecified flag" << endl;
    return -1;
  }
  // model and dict file name prefix
  string fprefix = string(prefix);
  string fout = string(fcontext);
  fout += ("." + flag + ".sample");
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
  Corpus tst = readData(fcontext, &d, false);

  // ----------------------------------------------
  // define model
  Model omodel, hmodel, rmodel;
  // only one of them is used in the following
  DCLMOutput<LSTMBuilder> olm(omodel, nlayers, inputdim, 
			      hiddendim, vocabsize);
  DCLMHidden<LSTMBuilder> hlm(hmodel, nlayers, inputdim, 
			      hiddendim, vocabsize);
  RNNLM<LSTMBuilder> rnnlm(rmodel, nlayers, inputdim,
			   hiddendim, vocabsize);
  // Load model
  cerr << "Load model from: " << fprefix << ".model" << endl;
  if (flag == "rnnlm"){
    load_model(fprefix, rmodel);
  } else if (flag == "output"){
    load_model(fprefix, omodel);
  } else if (flag == "hidden"){
    load_model(fprefix, hmodel);
  }

  // ---------------------------------------------
  // start testing
  double loss = 0;
  unsigned lines = 0, words = 0;
  //iterating over documents
  string sent; // generated sentence
  ComputationGraph cg;
  for (auto& context : tst){
    context.pop_back(); // remove the last sentence
    context.back().pop_back(); // remove the last token
    // get the right model
    if (flag == "hidden"){
      sent = hlm.RandomSample(context, cg, d);
    } else {
      cerr << "Unrecognized flag: " << flag << endl;
      return -1;
    }
    cerr << sent << endl;
    cerr << "===" << endl;    
    myfile << sent << "\n";
    myfile << "===" << "\n";
  }
  myfile.close();
  return 0;
}
