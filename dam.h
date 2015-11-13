#pragma once

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"

#include <iostream>

namespace cnn {

using namespace cnn::expr;

template <class Builder>
struct DocumentAttentionalModel {
  explicit DocumentAttentionalModel(Model& model, 
				    unsigned vocab_size, 
				    unsigned layers, 
				    unsigned embedding_dim, 
				    unsigned hidden_dim, 
				    unsigned align_dim);
  
  // forms a computation graph for the 
  Expression BuildGraph(const std::vector<std::vector<int>> &document, ComputationGraph& cg);
  
  LookupParameters* p_c;
  Parameters* p_R;
  Parameters* p_Q;
  Parameters* p_P;
  Parameters* p_bias;
  Parameters* p_Wa;
  Parameters* p_Ua;
  Parameters* p_va;
  Builder builder;
  unsigned context_dim;
  
  // statefull functions for incrementally creating computation graph, one
  // target word at a time
  void start_new_sentence(ComputationGraph &cg, bool first);
  Expression add_input(int tgt_tok, int t, ComputationGraph &cg);

  // state variables used in the above two methods
  Expression src;
  Expression i_R;
  Expression i_Q;
  Expression i_P;
  Expression i_bias;
  Expression i_Wa;
  Expression i_Ua;
  Expression i_va;
  Expression i_uax;
  Expression i_empty;
  std::vector<float> zeros;
  std::vector<Expression> context;
};
 
#define WTF(expression)							\
  std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression)					\
  std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression)			\
  WTF(expression)				\
    KTHXBYE(expression) 
 
 template <class Builder>
   DocumentAttentionalModel<Builder>::DocumentAttentionalModel(cnn::Model& model,
							       unsigned vocab_size, unsigned layers, unsigned embedding_dim, 
							       unsigned hidden_dim, unsigned align_dim) 
   : builder(layers, embedding_dim+layers*hidden_dim, hidden_dim, &model),
  context_dim(layers*hidden_dim)
    {
      p_c = model.add_lookup_parameters(vocab_size, {embedding_dim}); 
      p_R = model.add_parameters({vocab_size, hidden_dim});
      p_P = model.add_parameters({hidden_dim, embedding_dim});
      p_bias = model.add_parameters({vocab_size});
      p_Wa = model.add_parameters({align_dim, layers*hidden_dim});
      p_Ua = model.add_parameters({align_dim, context_dim});
      p_Q = model.add_parameters({hidden_dim, context_dim});
      p_va = model.add_parameters({align_dim});
    }
 
 template <class Builder>
   void DocumentAttentionalModel<Builder>::start_new_sentence(ComputationGraph &cg, bool first)
   {
     if (!first)
       context.push_back(concatenate(builder.final_h())); 
     builder.start_new_sequence();
     
     if (context.size() > 1) {
       src = concatenate_cols(context); 
       i_uax = i_Ua * src;
     }
   }
 
 template <class Builder>
   Expression DocumentAttentionalModel<Builder>::add_input(int tok, int t, ComputationGraph &cg)
   {
     Expression i_x_t = lookup(cg, p_c, tok);
     Expression i_c_t;
     if (context.size() > 1) {
       Expression i_wah_rep;
       if (t > 0) {
	 auto i_h_tm1 = concatenate(builder.final_h());
	 Expression i_wah = i_Wa * i_h_tm1;
	 i_wah_rep = concatenate_cols(std::vector<Expression>(context.size(), i_wah));
       }
       
       Expression i_e_t;
       if (t > 0) 
	 i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;
       else
	 i_e_t = transpose(tanh(i_uax)) * i_va;
       
       Expression i_alpha_t = softmax(i_e_t);
       i_c_t = src * i_alpha_t; 
     } else if (context.size() == 1) {
       i_c_t = context.back();
     } else {
       i_c_t = i_empty;
     }
     Expression input = concatenate(std::vector<Expression>({i_x_t, i_c_t})); 
     Expression i_y_t = builder.add_input(input);
     Expression i_tildet_t = tanh(affine_transform({i_y_t, i_Q, i_c_t, i_P, i_x_t}));
     Expression i_r_t = affine_transform({i_bias, i_R, i_tildet_t}); 
     
     return i_r_t;
   }
 
 template <class Builder>
   Expression DocumentAttentionalModel<Builder>::BuildGraph(const std::vector<std::vector<int>> &document, ComputationGraph& cg) 
   {
     builder.new_graph(cg);
     context.clear();
     
     i_R = parameter(cg, p_R); 
     i_Q = parameter(cg, p_Q);
     i_P = parameter(cg, p_P);
     i_bias = parameter(cg, p_bias);
     i_Wa = parameter(cg, p_Wa); 
     i_Ua = parameter(cg, p_Ua);
     i_va = parameter(cg, p_va);
     
     zeros.resize(context_dim, 0);
     i_empty = input(cg, {context_dim}, &zeros);
     
     std::vector<Expression> errs;
     bool first = true;
     for (const auto &sent: document) {
       start_new_sentence(cg, first);
       const unsigned tlen = sent.size() - 1; 
       for (unsigned t = 0; t < tlen; ++t) {
	 Expression i_r_t = add_input(sent[t], t, cg);
	 Expression i_err = pickneglogsoftmax(i_r_t, sent[t+1]);
	 errs.push_back(i_err);
       }
       first = false;
     }
     
     Expression i_nerr = sum(errs);
     return i_nerr;
   }
 
#undef WTF
#undef KTHXBYE
#undef LOLCAT
 
}; // namespace cnn
