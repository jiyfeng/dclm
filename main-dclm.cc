#include "util.hpp"
#include "training.hpp"
#include "test.hpp"
#include "sample.hpp"
#include <stdlib.h>

int NLAYERS = 2;

// main function
int main(int argc, char** argv) {
  
  // initialize cnn
  cnn::Initialize(argc, argv);
  // check arguments
  cout << "Number of arguments " << argc << endl;
  if (argc < 5) {
    cerr << "============================\n"
	 << "Usage: \n" 
	 << "\t" << argv[0] 
	 << " train train_file dev_file flag \n\t\t[input_dim] [hidden_dim] [learn_rate] [use_adagrad] [model_prefix]\n"
	 << "\t" << argv[0] 
	 << " test model_prefix test_file flag\n"
	 << "\t" << argv[0]
	 << " sample model_prefix test_file flag\n";
    return -1;
  }
  // parse command arguments
  string cmd = argv[1];
  if (cmd == "train"){
    cout << "Task: " << argv[1] <<endl;
    char* ftrn = argv[2];
    char* fdev = argv[3];
    string flag = string(argv[4]);
    unsigned inputdim = 16;
    unsigned hiddendim = 48;
    float lr0 = 0.1; // initial learning rate
    bool use_adagrad = false;
    string fmodel("");
    if (argc >= 6) inputdim = atoi(argv[5]);
    if (argc >= 7) hiddendim = atoi(argv[6]);
    if (argc >= 8) lr0 = atof(argv[7]);
    if (argc >= 9) use_adagrad = atoi(argv[8]);
    if (argc >= 10) fmodel = string(argv[9]);
    train(ftrn, fdev, NLAYERS, inputdim, hiddendim, 
	  flag, lr0, use_adagrad, fmodel);
  }
  else if(cmd == "test"){
    cout << "Task: "<< argv[1] << endl;
    char* prefix = argv[2];
    char* ftst = argv[3];
    string flag(argv[4]);
    test(ftst, prefix, flag);
  }
  else if(cmd == "sample"){
    cout << "Task: " << argv[1] << endl;
    char* prefix = argv[2];
    char* fcont = argv[3];
    string flag(argv[4]);
    randomsample(fcont, prefix, flag);
  }
  else{
    cerr << "Unrecognized command " << argv[1]<<endl;
  }
}
