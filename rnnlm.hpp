#ifndef RNNLM_HPP
#define RNNLM_HPP

#include "util.hpp"

template <class Builder>
class RNNLM{
private:
  LookupParameters* p_c; // word embeddings VxK1
  Parameters* p_R; // output layer: VxK2
  Parameters* p_R2; // forward context vector: VxK2
  Parameters* p_bias; // bias Vx1
  Parameters* p_context; // default context vector
  Builder builder;

public:
  RNNLM();
  RNNLM(Model& model, unsigned nlayers, 
	unsigned inputdim, unsigned hiddendim, 
	unsigned vocabsize):builder(nlayers, inputdim, 
				    hiddendim, &model){
    p_c = model.add_lookup_parameters(vocabsize, {inputdim}); 
    // for hidden output
    p_R = model.add_parameters({vocabsize, hiddendim});
    // for bias
    p_bias = model.add_parameters({vocabsize});
  }
  
  Expression BuildGraph(const Doc doc, ComputationGraph& cg){
    // reset RNN builder for new graph
    builder.new_graph(cg);  
    // define expression
    Expression i_R = parameter(cg, p_R);
    Expression i_bias = parameter(cg, p_bias);
    Expression i_x_t, i_h_t, i_y_t, i_err;
    // -----------------------------------------
    // build CG for the doc
    vector<Expression> errs;
    for (unsigned k = 0; k < doc.size(); k++){
      builder.start_new_sequence();
      // for each sentence in this doc
      auto sent = doc[k];
      unsigned slen = sent.size() - 1;
      // build RNN for the current sentence
      for (unsigned t = 0; t < slen; t++){
	// get word representation
	i_x_t = lookup(cg, p_c, sent[t]);
	// compute hidden state
	i_h_t = builder.add_input(i_x_t);
	// compute prediction
	i_y_t = (i_R * i_h_t) + i_bias;
	// get prediction error
	i_err = pickneglogsoftmax(i_y_t, sent[t+1]);
	// add back
	errs.push_back(i_err);
      }
    }
    Expression i_nerr = sum(errs);
    return i_nerr;
  }
};

#endif
