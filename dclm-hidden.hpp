#ifndef DCLM_HIDDEN_HPP
#define DCLM_HIDDEN_HPP

#include "util.hpp"

template <class Builder>
class DCLMHidden{
private:
  LookupParameters* p_c; // word embeddings VxK1
  Parameters* p_R; // output layer: VxK2
  Parameters* p_bias; // bias Vx1
  Parameters* p_context; // default context vector
  Parameters* p_transform; // transformation matrix
  Builder builder;

public:
  DCLMHidden();
  DCLMHidden(Model& model, unsigned nlayers, 
	     unsigned inputdim, unsigned hiddendim, 
	     unsigned vocabsize):builder(nlayers, 
					 inputdim+hiddendim, 
					 hiddendim, &model){
    p_c = model.add_lookup_parameters(vocabsize, {inputdim}); 
    // for hidden output
    p_R = model.add_parameters({vocabsize, hiddendim});
    // for bias
    p_bias = model.add_parameters({vocabsize});
    // for default context vector
    p_context = model.add_parameters({hiddendim});
  } // END of a constructor
  
  Expression BuildGraph(const Doc doc, ComputationGraph& cg){
    // reset RNN builder for new graph
    builder.new_graph(cg);  
    // define expression
    Expression i_R = parameter(cg, p_R);
    Expression i_bias = parameter(cg, p_bias);
    Expression i_context = parameter(cg, p_context);
    Expression cvec, i_x_t, i_h_t, i_y_t, i_err;
    vector<Expression> vec_exp;
    // ------------------------------------------
    // build CG for the doc
    vector<Expression> errs;
    for (unsigned k = 0; k < doc.size(); k++){
      // start a new sequence for each sentence
      builder.start_new_sequence();
      // for each sentence in this doc
      auto sent = doc[k];
      unsigned slen = sent.size() - 1;
      // get context vector if this is the first sent
      if (k == 0) cvec = i_context;
      // build RNN for the current sentence
      for (unsigned t = 0; t < slen; t++){
	// get word representation
	i_x_t = lookup(cg, p_c, sent[t]);
	vec_exp.clear();
	// add context vector
	vec_exp.push_back(i_x_t); 
	vec_exp.push_back(cvec);
	i_x_t = concatenate(vec_exp);
	// compute hidden state
	i_h_t = builder.add_input(i_x_t);
	// compute prediction
	i_y_t = (i_R * i_h_t) + i_bias;
	// get prediction error
	i_err = pickneglogsoftmax(i_y_t, sent[t+1]);
	// add back
	errs.push_back(i_err);
      }
      // update context vector
      cvec = i_h_t;
    }
    Expression i_nerr = sum(errs);
    return i_nerr;
  } // END of BuildGraph

  string RandomSample(const Doc cont, ComputationGraph& cg, 
		      cnn::Dict d, int max_len = 100){
    int kSOS = d.Convert("<s>");
    int kEOS = d.Convert("</s>");
    // define expression
    Expression i_R = parameter(cg, p_R);
    Expression i_bias = parameter(cg, p_bias);
    Expression i_context = parameter(cg, p_context);
    // Expression i_transform = parameter(cg, p_transform);
    Expression cvec, i_x_t, i_h_t, i_y_t, i_err;
    vector<Expression> vec_exp;
    // ------------------------------------------
    // build CG for the context
    ostringstream os;
    vector<string> conlist;
    conlist.push_back("but");
    conlist.push_back("so");
    for (auto& con : conlist){
      builder.new_graph(cg);
      for (unsigned k = 0; k < cont.size(); k++){
	// start a new sequence for each sentence
	builder.start_new_sequence();
	// for each sentence in this doc
	auto sent = cont[k];
	unsigned slen = sent.size() - 1;
	// get context vector if this is the first sent
	if (k == 0) cvec = i_context;
	// build RNN for the current sentence
	for (unsigned t = 0; t < slen; t++){
	  // get word representation
	  i_x_t = lookup(cg, p_c, sent[t]);
	  vec_exp.clear();
	  // add context vector
	  vec_exp.push_back(i_x_t); vec_exp.push_back(cvec);
	  i_x_t = concatenate(vec_exp);
	  // compute hidden state
	  i_h_t = builder.add_input(i_x_t);
	  // compute prediction
	  i_y_t = (i_R * i_h_t) + i_bias;
	  // get prediction error
	  // i_err = pickneglogsoftmax(i_y_t, sent[t+1]);
	  // add back
	  // errs.push_back(i_err);
	}
	// update context vector
	cvec = i_h_t;
      }
      // ------------------------------------------
      // random sampling word to form a sentence
      os << con << " ";
      int len = 0, cur = d.Convert(con);
      Expression ydist;
      while (len < max_len && cur != kEOS){
	len ++;
	// compute output prob
	i_x_t = lookup(cg, p_c, cur);
	vec_exp.clear();
	vec_exp.push_back(i_x_t);
	vec_exp.push_back(cvec);
	i_x_t = concatenate(vec_exp);
	i_h_t = builder.add_input(i_x_t);
	i_y_t = (i_R * i_h_t) + i_bias;
	ydist = softmax(i_y_t);
	// sample from prob
	unsigned w = 0;
	while (w == 0 || (int) w == kSOS){
	  auto dist = as_vector(cg.incremental_forward());
	  double p = rand01();
	  // cout << "kEOS = " << dist[kEOS] << endl;
	  for (; w < dist.size(); w++){
	    p -= dist[w];
	    if (p < 0.0) break;
	  }
	  if  (w == dist.size()) w = kEOS;
	}
	os << d.Convert(w) << " ";
	cur = w;
      }
      os << "\n";
    }
    // os << endl;
    string sample = os.str();
    return sample;
  } // END of RandomSample

};

#endif
