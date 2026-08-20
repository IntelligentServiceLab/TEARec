/*
<%
setup_pybind11(cfg)
%>
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <unordered_set>
#include <random>
#include <algorithm>

namespace py = pybind11;

class FastSampler {
private:
    std::random_device rd;
    std::mt19937 gen;

public:
    FastSampler(unsigned int seed = 2025) : gen(seed) {}

    // 快速负采样函数 - 避免重复检查
    std::vector<int> sample_neg_items_for_user(
        int user_id,
        const std::vector<int>& pos_items,
        int num_samples,
        int total_items) {

        std::vector<int> neg_items;
        std::unordered_set<int> pos_set(pos_items.begin(), pos_items.end());
        std::unordered_set<int> sampled_set;

        std::uniform_int_distribution<> dis(0, total_items - 1);

        while (neg_items.size() < num_samples) {
            int candidate = dis(gen);

            // 检查是否为正样本或已采样
            if (pos_set.find(candidate) == pos_set.end() &&
                sampled_set.find(candidate) == sampled_set.end()) {
                neg_items.push_back(candidate);
                sampled_set.insert(candidate);
            }
        }

        return neg_items;
    }

    // 批量负采样 - 一次性为多个用户采样
    std::vector<std::vector<int>> batch_sample_neg_items(
        const std::vector<int>& users,
        const std::vector<std::vector<int>>& users_pos_items,
        int num_samples_per_user,
        int total_items) {

        std::vector<std::vector<int>> batch_neg_items;
        batch_neg_items.reserve(users.size());

        for (size_t i = 0; i < users.size(); ++i) {
            auto neg_items = sample_neg_items_for_user(
                users[i], users_pos_items[i], num_samples_per_user, total_items);
            batch_neg_items.push_back(neg_items);
        }

        return batch_neg_items;
    }

    // 快速正采样函数
    std::vector<int> sample_pos_items_for_user(
        const std::vector<int>& pos_items,
        int num_samples) {

        if (num_samples >= pos_items.size()) {
            return pos_items;
        }

        std::vector<int> result;
        std::unordered_set<int> sampled_indices;
        std::uniform_int_distribution<> dis(0, pos_items.size() - 1);

        while (result.size() < num_samples) {
            int idx = dis(gen);
            if (sampled_indices.find(idx) == sampled_indices.end()) {
                result.push_back(pos_items[idx]);
                sampled_indices.insert(idx);
            }
        }

        return result;
    }

    // 采样用户
    std::vector<int> sample_users(
        const std::vector<int>& exist_users,
        int batch_size) {

        std::vector<int> users;

        if (batch_size <= exist_users.size()) {
            // 使用shuffle + resize来采样不重复的用户
            std::vector<int> temp_users = exist_users;
            std::shuffle(temp_users.begin(), temp_users.end(), gen);
            users.assign(temp_users.begin(), temp_users.begin() + batch_size);
        } else {
            // 如果需要的数量大于用户总数，允许重复采样
            std::uniform_int_distribution<> dis(0, exist_users.size() - 1);
            users.reserve(batch_size);
            for (int i = 0; i < batch_size; ++i) {
                users.push_back(exist_users[dis(gen)]);
            }
        }

        return users;
    }

    // 完整的批量采样函数
    py::tuple batch_sample(
        const std::vector<int>& exist_users,
        const std::vector<std::vector<int>>& train_items_dict,
        int batch_size,
        int total_items,
        int pos_samples_per_user = 1,
        int neg_samples_per_user = 1) {

        // 采样用户
        auto users = sample_users(exist_users, batch_size);

        std::vector<int> all_pos_items;
        std::vector<int> all_neg_items;

        // 为每个用户采样正负样本
        for (int user : users) {
            if (user < train_items_dict.size() && !train_items_dict[user].empty()) {
                // 采样正样本
                auto pos_items = sample_pos_items_for_user(
                    train_items_dict[user], pos_samples_per_user);
                all_pos_items.insert(all_pos_items.end(), pos_items.begin(), pos_items.end());

                // 采样负样本
                auto neg_items = sample_neg_items_for_user(
                    user, train_items_dict[user], neg_samples_per_user, total_items);
                all_neg_items.insert(all_neg_items.end(), neg_items.begin(), neg_items.end());
            }
        }

        return py::make_tuple(users, all_pos_items, all_neg_items);
    }
};

PYBIND11_MODULE(sample, m) {
    m.doc() = "TEARec fast sampling extension";

    py::class_<FastSampler>(m, "FastSampler")
        .def(py::init<unsigned int>(), py::arg("seed") = 2026)
        .def("sample_neg_items_for_user", &FastSampler::sample_neg_items_for_user,
             "Fast negative sampling for single user")
        .def("batch_sample_neg_items", &FastSampler::batch_sample_neg_items,
             "Batch negative sampling for multiple users")
        .def("sample_pos_items_for_user", &FastSampler::sample_pos_items_for_user,
             "Fast positive sampling for single user")
        .def("sample_users", &FastSampler::sample_users,
             "Sample users for batch")
        .def("batch_sample", &FastSampler::batch_sample,
             "Complete batch sampling function",
             py::arg("exist_users"), py::arg("train_items_dict"), py::arg("batch_size"),
             py::arg("total_items"), py::arg("pos_samples_per_user") = 1,
             py::arg("neg_samples_per_user") = 1);
}