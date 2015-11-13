#include "util.hpp"

// *******************************************************
// load model from a archive file
// *******************************************************
int load_model(string fname, Model& model){
  ifstream in(fname + ".model");
  boost::archive::text_iarchive ia(in);
  ia >> model;
  return 0;
}

// *******************************************************
// save model from a archive file
// *******************************************************
int save_model(string fname, Model& model){
  ofstream out(fname + ".model");
  boost::archive::text_oarchive oa(out);
  oa << model; 
  out.close();
  return 0;
}

// *******************************************************
// save dict from a archive file
// *******************************************************
int save_dict(string fname, cnn::Dict d){
  fname += ".dict";
  ofstream out(fname);
  boost::archive::text_oarchive odict(out);
  odict << d; out.close();
  return 0;
}

// *******************************************************
// load dict from a archive file
// *******************************************************
int load_dict(string fname, cnn::Dict& d){
  fname += ".dict";
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> d; in.close();
  return 0;
}

// *******************************************************
// read sentences and convect tokens to indices
// *******************************************************
Sent MyReadSentence(const std::string& line, 
		    Dict* sd, 
		    bool update) {
  vector<string> strs;
  // boost::split(strs, line, boost::is_any_of(" "));
  istringstream in(line);
  string word;
  Sent res;
  res.push_back(sd->Convert("<s>"));
  // for (auto& word : strs){
  while (in){
    in >> word;
    if (word.empty()) break;
    // cerr << "word = " << word << endl;
    if (update){
      res.push_back(sd->Convert(word));
    } else {
      if (sd->Contains(word)){
	res.push_back(sd->Convert(word));
      }else{
	res.push_back(sd->Convert("UNK"));
      }
    }
  }
  res.push_back(sd->Convert("</s>"));
  return res;
}

// *****************************************************
// 
// *****************************************************
Doc makeDoc(){
  vector<vector<int>> doc;
  return doc;
}

// *****************************************************
// read training and dev data
// *****************************************************
Corpus readData(char* filename, 
		cnn::Dict* dptr,
		bool b_update){
  cerr << "reading data from "<< filename << endl;
  Corpus corpus;
  Doc doc;
  Sent sent;
  string line;
  int tlc = 0;
  int toks = 0;
  ifstream in(filename);
  while(getline(in, line)){
    ++tlc;
    if (line[0] != '='){
      sent = MyReadSentence(line, dptr, b_update);
      if (sent.size() > 0){
	doc.push_back(sent);
	toks += doc.back().size();
      } else {
	cerr << "Empty sentence: " << line << endl;
      }
    } else {
      if (doc.size() > 0){
	corpus.push_back(doc);
	doc = makeDoc();
      } else {
	cerr << "Empty document " << endl;
      }
    }
  }
  if (doc.size() > 0){
    corpus.push_back(doc);
  }
  cerr << corpus.size() << " docs, " << tlc << " lines, " 
       << toks << " tokens, " << dptr->size() 
       << " types." << endl;
  return(corpus);
}

// ******************************************************
// Convert 1-D tensor to vector<float>
// so we can create an expression for it
// ******************************************************
vector<float> convertT2V(const Tensor& t){
  vector<float> vf;
  int dim = t.d.d[0];
  for (int idx = 0; idx < dim; idx++){
    vf.push_back(t.v[idx]);
  }
  return vf;
}

// ******************************************************
// Check the directory, if doesn't exist, create one
// ******************************************************
int check_dir(string path){
  boost::filesystem::path dir(path);
  if(!(boost::filesystem::exists(dir))){
    if (boost::filesystem::create_directory(dir)){
      std::cout << "....Successfully Created !" << "\n";
    }
  }
}

// ******************************************************
// Segment a long document into several short ones
// ******************************************************
Corpus segment_doc(Corpus corpus, int thresh){
  Corpus newcorpus;
  for (auto& doc : corpus){
    if (doc.size() <= thresh){
      newcorpus.push_back(doc);
      continue;
    }
    Doc tmpdoc;
    int counter = 0;
    for (auto& sent : doc){
      if (counter <= thresh){
	tmpdoc.push_back(sent);
	counter ++;
      } else {
	newcorpus.push_back(tmpdoc);
	tmpdoc.clear();
	tmpdoc.push_back(sent);
	counter = 1;
      }
    }
    if (tmpdoc.size() > 0){
      newcorpus.push_back(tmpdoc);
      tmpdoc.clear();
    }
  }
  return newcorpus;
}
