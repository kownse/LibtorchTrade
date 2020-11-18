#include "Agent.h"
#include "AttGru.h"
#include "../TrainOption.h"

namespace train::agent {

Agent* Agent::create(const TrainOption &option) {
    if (option.agentType == "AttGru") {
        return new AttGru(option);
    } else {
        std::cout << "Unsupported agent type: " << option.agentType << std::endl;
    }
    return nullptr;
}

void Agent::saveToFile(std::string path) {
    torch::serialize::OutputArchive output_archive;
    save(output_archive);
    output_archive.save_to(path);
}

void Agent::loadFromFile(std::string path) {
    torch::serialize::InputArchive input_archive;
    input_archive.load_from(path);
    load(input_archive);
}

};