#ifndef UTIL_HPP
#define UTIL_HPP

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/tensor.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <thread>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cnn;

// ********************************************************
// Predefined information, used for the entire project
// ********************************************************
// Redefined types
typedef vector<int> Sent;
typedef vector<Sent> Doc;
typedef vector<Doc> Corpus;

// *******************************************************
// load model from a archive file
// *******************************************************
int load_model(string fname, Model& model);

// *******************************************************
// save model from a archive file
// *******************************************************
int save_model(string fname, Model& model);

// *******************************************************
// save dict from a archive file
// *******************************************************
int save_dict(string fname, cnn::Dict d);

// *******************************************************
// load dict from a archive file
// *******************************************************
int load_dict(string fname, cnn::Dict& d);

// *******************************************************
// read sentences and convect tokens to indices
// *******************************************************
Sent MyReadSentence(const std::string& line, 
		    Dict* sd, 
		    bool update);

// *****************************************************
// 
// *****************************************************
Doc makeDoc();

// *****************************************************
// read training and dev data
// *****************************************************
Corpus readData(char* filename, 
		cnn::Dict* dptr,
		bool b_update = true);


// ******************************************************
// Convert 1-D tensor to vector<float>
// so we can create an expression for it
// ******************************************************
vector<float> convertT2V(const Tensor& t);

// ******************************************************
// Check the directory, if doesn't exist, create one
// ******************************************************
int check_dir(string path);

// ******************************************************
// Segment a long document into several short ones
// ******************************************************
Corpus segment_doc(Corpus doc, int thresh);

#endif
