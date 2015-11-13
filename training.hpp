#ifndef TRAINING_HPP
#define TRAINING_HPP

#include "dclm-output.hpp"
#include "dclm-hidden.hpp"
#include "rnnlm.hpp"
#include "hrnnlm.hpp"
#include "util.hpp"

int train(char* ftrn, char* fdev, unsigned nlayers = 2, 
	  unsigned inputdim = 16, unsigned hiddendim = 48, 
	  string flag = "output", float lr0 = 0.1, 
	  bool use_adagrad = false, string fmodel = "");

#endif
