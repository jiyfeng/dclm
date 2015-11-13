#ifndef DCLM_OUTPUT_HPP
#define DCLM_OUTPUT_HPP

#include "util.hpp"

template <class Builder>
class DCLMOutput{
private:
  LookupParameters* p_c; // word embeddings VxK1
  Parameters* p_R; // output layer: VxK2
  Parameters* p_R2; // forward context vector: VxK2
  Parameters* p_bias; // bias Vx1
  Parameters* p_context; // default context vector for sent-level
  Builder builder;

public:
  DCLMOutput();
  DCLMOutput(Model& model, unsigned nlayers, 
	     unsigned inputdim, unsigned hiddendim, 
	     unsigned vocabsize):builder(nlayers, inputdim, 
					 hiddendim, &model){
    p_c = model.add_lookup_parameters(vocabsize, {inputdim}); 
    // for hidden output
    p_R = model.add_parameters({vocabsize, hiddendim});
    // for forward context vector
    p_R2 = model.add_parameters({vocabsize, hiddendim});
    // for bias
    p_bias = model.add_parameters({vocabsize});
    // for default context vector
    p_context = model.add_parameters({hiddendim});
  }
  
  Expression BuildGraph(const Doc doc, ComputationGraph& cg){
    // reset RNN builder for new graph
    builder.new_graph(cg);  
    // define expression
    Expression i_R = parameter(cg, p_R);
    Expression i_R2 = parameter(cg, p_R2);
    Expression i_bias = parameter(cg, p_bias);
    Expression i_context = parameter(cg, p_context);
    Expression cvec, i_x_t, i_h_t, i_y_t, i_err, ccpb;
    // -----------------------------------------
    // build CG for the doc
    vector<Expression> errs;
    for (unsigned k = 0; k < doc.size(); k++){
      builder.start_new_sequence();
      // for each sentence in this doc
      auto sent = doc[k];
      unsigned slen = sent.size() - 1;
      // start a new sequence for each sentence
      if (k == 0) cvec = i_context;
      // build RNN for the current sentence
      ccpb = (i_R2 * cvec) + i_bias;
      for (unsigned t = 0; t < slen; t++){
	// get word representation
	i_x_t = lookup(cg, p_c, sent[t]);
	// compute hidden state
	i_h_t = builder.add_input(i_x_t);
	// compute prediction
	i_y_t = (i_R * i_h_t) + ccpb;
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
  }
};

#endif
