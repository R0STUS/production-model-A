#include "main.h"

struct Neuron {
    std::vector<double> weights;
    double bias = 0.0;
};

double activation(double x) {
    if (std::isfinite(x)) {
        return x;
    }
    else {
        return 0.0;
    }
}

double output(const std::vector<double>& inputs, const std::vector<Neuron>& network) {
    double sum = 0.0;
    for (size_t i = 0; i < network.size(); i++) {
        double dot_product = 0.0;
        for (size_t j = 0; j < inputs.size(); j++) {
            dot_product += inputs[j] * network[i].weights[j];
        }
        sum += activation(dot_product + network[i].bias);
    }
    return sum;
}

void train(const std::vector<std::vector<double>>& inputs, const std::vector<double>& targets, std::vector<Neuron>& network, double learning_rate, int epochs) {
    std::vector<double> layer1(network.size() / 2);
    std::vector<double> layer2(network.size() - network.size() / 2);
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < inputs.size(); i++) {
            double out = output(inputs[i], network);
            double error = targets[i] - out;
            for (int j = network.size() - 1; j >= 0; j--) {
                for (int k = 0; k < (j < network.size() / 2 ? inputs[i].size() : network.size() / 2); k++) {
                    network[j].weights[k] += learning_rate * error * (j < network.size() / 2 ? inputs[i][k] : layer1[k]);
                }
                network[j].bias += learning_rate * error;
            }
        }
    }
}

void save_network(const std::vector<Neuron>& network, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        for (const auto& neuron : network) {
            file.write(reinterpret_cast<const char*>(&neuron.weights[0]), sizeof(double) * neuron.weights.size());
            file.write(reinterpret_cast<const char*>(&neuron.bias), sizeof(double));
        }
        file.close();
    }
    else {
        std::cout << "Unable to open file: " << filename << std::endl;
    }
}

void load_network(std::vector<Neuron>& network, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        for (auto& neuron : network) {
            file.read(reinterpret_cast<char*>(&neuron.weights[0]), sizeof(double) * neuron.weights.size());
            file.read(reinterpret_cast<char*>(&neuron.bias), sizeof(double));
        }
        file.close();
    }
    else {
        std::cout << "Unable to open file: " << filename << std::endl;
    }
}

int main() {
    std::vector<Neuron> network(6);
    for (auto& neuron : network) {
        neuron.weights = std::vector<double>(3, 0.5);
    }

    load_network(network, "network.bin");

    std::vector<double> inputs;
    double input;
    float target;
    std::deque<double> input_history;
    const int history_size = 5;

    while (true) {
        std::cout << "Enter 3 numbers: ";
        for (int i = 0; i < 3; i++) {
            std::cin >> input;
            if (std::cin.fail()) {
                std::cerr << "Invalid input. Please enter a number." << std::endl;
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                i--;

                continue;
            }
            inputs.push_back(input);
        }

        for (double x : inputs) {
            if (!std::isfinite(x)) {
                std::cerr << "Invalid input. Please enter finite numbers." << std::endl;
                inputs.clear();
                break;
            }
        }

        if (inputs.empty()) {
            continue;
        }

        std::vector<std::vector<double>> input_vector = { inputs };
        double predicted_value = output(inputs, network);
        std::cout << "Predicted 4th value: " << int(predicted_value) << std::endl;

        std::cout << "Enter the 4th value: ";
        std::cin >> input;
        int real = int(input);
        if (std::cin.fail()) {
            std::cerr << "Invalid input. Please enter a number." << std::endl;
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }

        std::vector<double> target_vector = { input };
        train(input_vector, target_vector, network, 0.01, 100000);

        save_network(network, "network.bin");

        std::cout << "Rate:  ";
        if (int(predicted_value) == real) {
            target = 1;
        }
        else {
            target = -1;
        }
        std::cout << target << std::endl;
        if (std::cin.fail()) {
            std::cerr << "Invalid input. Please enter a -1, 0 or 1 value." << std::endl;
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }

        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        inputs.clear();
    }

    return 0;
}

