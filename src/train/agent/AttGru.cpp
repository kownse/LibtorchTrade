#include "AttGru.h"
#include "../TrainOption.h"
// #include <memory>

using torch::indexing::Slice;
using torch::indexing::None;

namespace train::agent {

AttGru::AttGru(const train::TrainOption &option)
    : Agent(1),
    attn(option.cols.size(), option.cols.size(), option.drop),
    gru(torch::nn::GRUOptions(option.cols.size(), option.rnnBase).num_layers(option.rnnLayers)),
    fc_s_1(option.rnnBase, option.linearBase),
    fc_s_2(option.linearBase, option.linearBase / 2),
    fc_s_out(option.linearBase / 2, 1),
    fc_pg_1(option.rnnBase, option.linearBase),
    fc_pg_2(option.linearBase, option.linearBase / 2),
    fc_pg_out(option.linearBase / 2, 2),
    norm_gru(option.rnnBase),
    norm_fc_1(option.linearBase),
    norm_fc_2(option.linearBase / 2),
    norm_pg_1(option.linearBase ),
    norm_pg_2(option.linearBase / 2),
    dropout(option.drop)
{

    // register_module("f_in", f_in);
    register_module<Attention>("attn", std::make_shared<Attention>(attn));

    register_module("gru", gru);
    register_module("norm_gru", norm_gru);

    register_module("fc_s_1", fc_s_1);
    register_module("norm_fc_1", norm_fc_1);
    register_module("fc_s_2", fc_s_2);
    register_module("norm_fc_2", norm_fc_2);
    register_module("fc_s_out", fc_s_out);

    register_module("fc_pg_1", fc_pg_1);
    register_module("norm_pg_1", norm_pg_1);
    register_module("fc_pg_2", fc_pg_2);
    register_module("norm_pg_2", norm_pg_2);
    register_module("fc_pg_out", fc_pg_out);
}

std::vector<torch::Tensor> AttGru::forward(std::vector<torch::Tensor> inputs) {
    auto o_t = attn.forward(inputs[0]);

    auto out = gru->forward(o_t, inputs[1]);
    o_t = norm_gru(std::get<0>(out).index({0}));
    auto h = std::get<1>(out);


    auto action = torch::relu(norm_pg_1(fc_pg_1->forward(o_t)));
    action = torch::relu(norm_pg_2(fc_pg_2->forward(action)));
    action = torch::dropout(action, /*p=*/dropout, /*train=*/is_training());
    action = torch::relu(fc_pg_out->forward(action));
    action = torch::softmax(action, {-1});

    auto value = torch::relu(norm_fc_1(fc_s_1->forward(o_t)));
    value = torch::relu(norm_fc_2(fc_s_2->forward(value)));
    value = torch::dropout(value, /*p=*/dropout, /*train=*/is_training());
    value = fc_s_out->forward(value);

    return {action.index({Slice(), 1}), value, h};
}

};