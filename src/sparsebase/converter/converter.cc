#include "converter.h"
#ifdef USE_CUDA
#include "sparsebase/converter/converter_cuda.cuh"
#include "sparsebase/format/cuda_csr_cuda.cuh"
#endif

#include <algorithm>
#include <deque>
#include <unordered_set>

#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"

#include "sparsebase/utils/utils.h"
#include "sparsebase/utils/logger.h"
#include "sparsebase/utils/exception.h"


namespace sparsebase::converter {

ConversionMap *Converter::get_conversion_map(bool is_move_conversion) {
  if (is_move_conversion)
    return &move_conversion_map_;
  else
    return &copy_conversion_map_;
}

const ConversionMap * Converter::get_conversion_map(bool is_move_conversion) const {
  if (is_move_conversion)
    return &move_conversion_map_;
  else
    return &copy_conversion_map_;
}


void Converter::RegisterConversionFunction(std::type_index from_type,
                                           std::type_index to_type,
                                           ConversionFunction conv_func,
                                           ConversionCondition edge_condition,
                                           bool is_move_conversion) {
  auto map = get_conversion_map(is_move_conversion);
  if (map->count(from_type) == 0) {
    map->emplace(
        from_type,
        std::unordered_map<std::type_index,
                           std::vector<std::tuple<ConversionCondition,
                                                  ConversionFunction>>>());
  }

  if ((*map)[from_type].count(to_type) == 0) {
    (*map)[from_type][to_type].push_back(
        std::make_tuple<ConversionCondition, ConversionFunction>(
            std::forward<ConversionCondition>(edge_condition),
            std::forward<ConversionFunction>(conv_func)));
    //(*map)[from_type].emplace(to_type, {make_tuple<EdgeConditional,
    // ConversionFunction>(std::forward<EdgeConditional>(edge_condition),
    // std::forward<ConversionFunction>(conv_func))});
  } else {
    (*map)[from_type][to_type].push_back(
        std::make_tuple<ConversionCondition, ConversionFunction>(
            std::forward<ConversionCondition>(edge_condition),
            std::forward<ConversionFunction>(conv_func)));
  }
}

format::Format *Converter::Convert(format::Format *source, std::type_index to_type,
                           std::vector<context::Context *> to_contexts,
                           bool is_move_conversion) const {
  auto outputs =
      ConvertCached(source, to_type, to_contexts, is_move_conversion);
  if (outputs.size() > 1)
    std::transform(outputs.begin(), outputs.end() - 1, outputs.begin(),
                   [](format::Format *f) {
                     delete f;
                     return nullptr;
                   });
  return outputs.back();
}

std::vector<format::Format *> Converter::ConvertCached (
    format::Format *source, std::type_index to_type,
    std::vector<context::Context *> to_contexts, bool is_move_conversion) const {
  if (to_type == source->get_id() &&
      std::find_if(to_contexts.begin(), to_contexts.end(),
                   [&source](context::Context *from) {
                     return from->IsEquivalent(source->get_context());
                   }) != to_contexts.end()) {
    return {source};
  }

  ConversionChain chain =
      GetConversionChain(source->get_id(), source->get_context(),
                         to_type, to_contexts, is_move_conversion);
  if (!chain)
    throw utils::ConversionException(source->get_name(),
                              utils::demangle(to_type));
  auto outputs = ApplyConversionChain(chain, source, false);
  return std::vector<format::Format *>(outputs.begin() + 1, outputs.end());
}

format::Format *Converter::Convert(format::Format *source, std::type_index to_type,
                           context::Context *to_context,
                           bool is_move_conversion) const {
  return Convert(source, to_type, std::vector<context::Context *>({to_context}),
                 is_move_conversion);
}

std::vector<format::Format *> Converter::ConvertCached(format::Format *source,
                                               std::type_index to_type,
                                               context::Context *to_context,
                                               bool is_move_conversion) const {
  return ConvertCached(source, to_type,
                       std::vector<context::Context *>({to_context}),
                       is_move_conversion);
}

/*
std::optional<std::tuple<Converter::CostType, context::Context*>>
Converter::CanDirectlyConvert(std::type_index from_type, context::Context*
from_context, std::type_index to_type, const std::vector<context::Context *>
&to_contexts, bool is_move_conversion){ auto map =
get_conversion_map(is_move_conversion); if (map->find(from_type) != map->end())
{ if ((*map)[from_type].find(to_type) != (*map)[from_type].end()) { for (auto
condition_function_pair : (*map)[from_type][to_type]) { for (auto to_context :
to_contexts) { if (std::get<0>(condition_function_pair)(from_context,
to_context)) { return std::make_tuple<bool, context::Context *>( true,
std::forward<context::Context *>(to_context));
          }
        }
      }
    }
  }
  return {};
}
 */

std::vector<ConversionStep> Converter::ConversionBFS(
    std::type_index from_type, context::Context *from_context,
    std::type_index to_type, const std::vector<context::Context *> &to_contexts,
    const ConversionMap * map) {
  std::deque<std::type_index> frontier{from_type};
  std::unordered_map<std::type_index,
                     std::pair<std::type_index, ConversionStep>>
      seen;
  seen.emplace(from_type,
               std::make_pair(from_type, std::make_tuple(nullptr, nullptr, 0)));
  int level = 1;
  size_t level_size = 1;
  while (!frontier.empty()) {
    for (int i = 0; i < level_size; i++) {
      auto curr = frontier.back();
      frontier.pop_back();
      if (map->find(curr) != map->end())
        for (const auto &neighbor : map->at(curr)) {
          if (seen.find(neighbor.first) == seen.end()) {
            for (auto curr_to_neighbor_functions :
                neighbor.second) {  // go over every edge curr->neighbor
              bool found_an_edge = false;
              for (auto to_context : to_contexts) {  // check if, with the given
                                                    // contexts, this edge exists
                if (std::get<0>(curr_to_neighbor_functions)(from_context,
                                                            to_context)) {
                  seen.emplace(
                      neighbor.first,
                      std::make_pair(
                          curr,
                          std::make_tuple(std::get<1>(curr_to_neighbor_functions),
                                          to_context, 1)));
                  frontier.emplace_back(neighbor.first);
                  if (neighbor.first == to_type) {
                    std::vector<ConversionStep> output(level);
                    std::type_index tip = to_type;
                    for (int j = level - 1; j >= 0; j--) {
                      output[j] = seen.at(tip).second;
                      tip = seen.at(tip).first;
                    }
                    return output;
                  }
                  found_an_edge = true;
                  break;  // no need to check other contexts
                }
              }
              if (found_an_edge)
                break;  // no need to check other edges between curr->neighbor
            }
          }
        }
    }
    level++;
    level_size = frontier.size();
  }
  return {};
}

ConversionChain Converter::GetConversionChain(
    std::type_index from_type, context::Context *from_context,
    std::type_index to_type, const std::vector<context::Context *> &to_contexts,
    bool is_move_conversion) const {
  // If the source doesn't need conversion, return an empty but existing
  // optional
  if (from_type == to_type && std::find(to_contexts.begin(), to_contexts.end(),
                                        from_context) != to_contexts.end())
    return ConversionChain(std::in_place);
  auto map = get_conversion_map(is_move_conversion);
  auto conversions =
      ConversionBFS(from_type, from_context, to_type, to_contexts, map);
  if (!conversions.empty())
    return std::make_tuple(conversions, conversions.size());
  else
    return {};
}
bool Converter::CanConvert(std::type_index from_type,
                           context::Context *from_context,
                           std::type_index to_type,
                           context::Context *to_context,
                           bool is_move_conversion) const {
  return GetConversionChain(from_type, from_context, to_type, {to_context},
                            is_move_conversion)
      .has_value();
}

bool Converter::CanConvert(std::type_index from_type,
                           context::Context *from_context,
                           std::type_index to_type,
                           const std::vector<context::Context *> &to_contexts,
                           bool is_move_conversion) {
  return GetConversionChain(from_type, from_context, to_type, to_contexts,
                            is_move_conversion)
      .has_value();
}

void Converter::ClearConversionFunctions(std::type_index from_type,
                                         std::type_index to_type,
                                         bool move_conversion) {
  auto map = get_conversion_map(move_conversion);
  if (map->find(from_type) != map->end()) {
    if ((*map)[from_type].find(to_type) != (*map)[from_type].end()) {
      (*map)[from_type].erase(to_type);
      if ((*map)[from_type].size() == 0) map->erase(from_type);
    }
  }
}

/*! Removes all conversion functions from the current converter
 */
void Converter::ClearConversionFunctions(bool move_conversion) {
  auto map = get_conversion_map(move_conversion);
  map->clear();
}

std::vector<format::Format *> Converter::ApplyConversionChain(
    const ConversionChain &chain, format::Format *input,
    bool clear_intermediate) {
  std::vector<format::Format *> format_chain{input};
  if (chain) {
    auto conversion_chain = std::get<0>(*chain);
    format::Format *current_format = input;
    for (int i = 0; i < conversion_chain.size(); i++) {
      const auto &conversion_step = conversion_chain[i];
      auto *res = std::get<0>(conversion_step)(current_format,
                                               std::get<1>(conversion_step));
      if (!clear_intermediate || i == conversion_chain.size() - 1)
        format_chain.push_back(res);
      else if (i != 0)
        delete current_format;
      current_format = res;
    }
  }
  return format_chain;
}

std::vector<std::vector<format::Format *>> Converter::ApplyConversionSchema(
    const ConversionSchema &cs, const std::vector<format::Format *> &packed_sfs,
    bool clear_intermediate) {
  // Each element in ret is a chain of formats resulting from
  // converting one of the packed_sfs
  std::vector<std::vector<format::Format *>> ret;
  for (int i = 0; i < cs.size(); i++) {
    const auto &conversion = cs[i];
    std::vector<format::Format *> format_chain =
        ApplyConversionChain(conversion, packed_sfs[i], clear_intermediate);
    ret.push_back(format_chain);
  }
  return ret;
}
Converter::~Converter() {}

}  // namespace sparsebase::converter
