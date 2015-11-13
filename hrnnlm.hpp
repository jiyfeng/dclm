#ifndef HRNNLM_HPP
#define HRNNLM_HPP

#include "util.hpp"

template <class Builder>
class HRNNLM{
private:
  LookupParameters* p_c; 
  // LookupParameters* p_c2;
  Parameters* p_R; 
  Parameters* p_R2; 
  Parameters* p_context;
  Parameters* p_bias;
  Parameters* p_bias2;
  Builder sbuilder, wbuilder;
  unsigned hd;
  vector<vector<float>> stensor;

public:
  HRNNLM();
  HRNNLM(Model& smodel, Model& wmodel, unsigned nlayers, 
	 unsigned inputdim, unsigned hiddendim, 
	 unsigned vocabsize){
    hd = hiddendim;
    // word-level builder
    wbuilder = Builder(nlayers, inputdim+hiddendim, 
		       hiddendim, &wmodel);
    // word embedding for word level
    p_c = wmodel.add_lookup_parameters(vocabsize, {inputdim}); 
    // word-level output weight metrix
    p_R = wmodel.add_parameters({vocabsize, hiddendim});
    // word-level bias term
    p_bias = wmodel.add_parameters({vocabsize});
    // default context vector
    p_context = wmodel.add_parameters({hiddendim});
    // sentence-level builder
    sbuilder = Builder(nlayers, inputdim, hiddendim, &smodel);
    // word embedding for sentence level
    // p_c2 = smodel.add_lookup_parameters(vocabsize, {inputdim});
    // sentence-level output weight matrix
    p_R2 = smodel.add_parameters({vocabsize, hiddendim});
    // sentence-level bias term
    p_bias2 = smodel.add_parameters({vocabsize});
  }

  Expression BuildWordGraph(const Doc doc, ComputationGraph& cg){
    // reset RNN builder
    wbuilder.new_graph(cg);
    // define expression
    Expression i_R = parameter(cg, p_R);
    Expression i_bias = parameter(cg, p_bias);
    Expression i_context = parameter(cg, p_context);
    vector<Expression> errs, vec_exp, sentexp;
    Expression i_x_t, i_h_t, i_y_t, i_err, cvec;
    // start building rnn
    for (unsigned k = 0; k < doc.size(); k++){
      wbuilder.start_new_sequence();
      auto sent = doc[k];
      // Get context representation (sentence-level 
      //  hidden state from s-level LM
      if (k == 0){
      	cvec = i_context;
      } else {
      	cvec = input(cg, {(unsigned)stensor[k-1].size()}, 
		     &(stensor[k-1]));
      }
      // build word-level rnnlm
      unsigned slen = sent.size() - 1;
      for (unsigned t = 0; t < slen; t++){
	// get word representation
	i_x_t = lookup(cg, p_c, sent[t]);
	vec_exp.clear();
	vec_exp.push_back(i_x_t);
	vec_exp.push_back(cvec);
	i_x_t = concatenate(vec_exp);
	// compute hidden state
	i_h_t = wbuilder.add_input(i_x_t);
	i_y_t = (i_R * i_h_t);
	// compute prediction error
	i_err = pickneglogsoftmax(i_y_t, sent[t+1]);
	// cerr << as_scalar(i_err.value()) << " ";
	errs.push_back(i_err);
      }
      // cerr << endl;
    }
    Expression i_nerr = sum(errs);
    return i_nerr;
  }
  
  Expression BuildSentGraph(const Doc doc, ComputationGraph& cg){
    // reset RNN builder for new graph
    stensor.clear();
    sbuilder.new_graph(cg);  
    sbuilder.start_new_sequence();
    // define expression
    Expression i_R2 = parameter(cg, p_R2);
    // -----------------------------------------
    // build sentence level language model for oen doc
    Expression i_x_t, i_h_t, i_y_t, i_err;
    vector<Expression> errs;
    for (unsigned k = 0; k < doc.size()-1; k++){
      auto& sent = doc[k];
      // ---------------------------------------
      // predict words in the sentence
      // ignore the first and last token, as they 
      //   are <s> and </s>
      i_x_t = const_lookup(cg, p_c, sent[1]);
      for (unsigned t = 2; t < sent.size() - 1; t++){
      	// accumulate word representation
      	//   don't update word embeddings
      	i_x_t = i_x_t + const_lookup(cg, p_c, sent[t]);
      }
      // compute hidden state
      i_h_t = sbuilder.add_input(i_x_t);
      // compute prediction for every words in this sent
      i_y_t = i_R2 * i_h_t;
      // get next sentence
      // cerr << "pick neg log softmax " << endl;
      auto& nextsent = doc[k+1];
      i_err = pickneglogsoftmax(i_y_t, nextsent[1]);
      errs.push_back(i_err);
      for (unsigned t = 1; t < nextsent.size() - 1; t++){
      	// compute predict err
      	i_err = pickneglogsoftmax(i_y_t, nextsent[t]);
      	// store
      	errs.push_back(i_err);
      }
      cg.incremental_forward();
      vector<float> vf = convertT2V(i_h_t.value());
      stensor.push_back(vf);
    }
    // sum over all errors for updating
    Expression i_nerr = sum(errs);
    return i_nerr;
  }

};

#endif
