#ifndef SPARSEBASE_PROJECT_FUNCTIONMATCHERMIXIN_H
#define SPARSEBASE_PROJECT_FUNCTIONMATCHERMIXIN_H

#include "sparsebase/config.h"
#include "parameterizable.h"
#include "utils.h"
#include "converter/converter.h"
#include <vector>

namespace sparsebase::utils {

//! Template for implementation functions of all preprocesses
/*!
  \tparam ReturnType the return type of preprocessing functions
  \param formats a vector of pointers at format::Format objects
  \param params a polymorphic pointer at a Parameters object
*/
template <typename ReturnType>
using PreprocessFunction = ReturnType (*)(std::vector<format::Format *> formats,
                                          utils::Parameters *params);

//! A mixin that attaches the functionality of matching keys to functions
/*!
  This mixin attaches the functionality of matching keys (which, by default, are
  vectors of type indices) to function pointer objects (by default, their
  signature is PreprocessFunction). \tparam ReturnType the return type that will
  be returned by the preprocessing function implementations \tparam Function the
  function signatures that keys will map to. Default is
  sparsebase::preprocess::PreprocessFunction \tparam Key the type of the keys
  used to access function in the inner maps. Default is
  std::vector<std::type_index>> \tparam KeyHash the hash function used to has
  keys. \tparam KeyEqualTo the function used to evaluate equality of keys
*/
template <typename ReturnType,
    class PreprocessingImpl = Parameterizable,
    typename Function = PreprocessFunction<ReturnType>,
    typename Key = std::vector<std::type_index>,
    typename KeyHash = TypeIndexVectorHash,
    typename KeyEqualTo = std::equal_to<std::vector<std::type_index>>>
class FunctionMatcherMixin : public PreprocessingImpl {
  //! Defines a map between `Key` objects and function pointer `Function`
  //! objects.
  typedef std::unordered_map<Key, Function, KeyHash, KeyEqualTo> ConversionMap;

 public:
  std::vector<Key> GetAvailableFormats() {
    std::vector<Key> keys;
    for (auto element : map_to_function_) {
      keys.push_back(element.first);
    }
    return keys;
  }
  //! Register a key to a function as long as that key isn't already registered
  /*!
    \param key_of_function key used in the map
    \param func_ptr function pointer being registered
    \return True if the function was registered successfully and false otherwise
  */
  bool RegisterFunctionNoOverride(const Key &key_of_function,
                                  const Function &func_ptr);
  //! Register a key to a function and overrides previous registered function
  //! (if any)
  /*!
    \param key_of_function key used in the map
    \param func_ptr function pointer being registered
  */
  void RegisterFunction(const Key &key_of_function, const Function &func_ptr);
  //! Unregister a key from the map if the key was registered to a function
  /*!
    \param key_of_function key to unregister
    \return true if the key was unregistered successfully, and false if it
    wasn't already registerd to something.
  */
  bool UnregisterFunction(const Key &key_of_function);

 protected:
  using PreprocessingImpl::PreprocessingImpl;
  //! Map between `Key` objects and function pointer `Function` objects.
  ConversionMap map_to_function_;
  //! Determines the exact Function and format conversions needed to carry out
  //! preprocessing
  /*!
   * \param packed_formats a vector of the input Format* needed for conversion.
   * \param key the Key representing the input formats.
   * \param map the map between Keys and Functions used to find the needed
   * function. \param contexts Contexts available for execution of the
   * preprocessing. \return a tuple of a) the Function to use,
   * and b) a converter::ConversionSchemaConditional indicating
   * conversions to be done on input Format objects.
   */
  std::tuple<Function, converter::ConversionSchema> GetFunction(
      std::vector<format::Format *> packed_formats, Key key, ConversionMap map,
      std::vector<context::Context *> contexts);
  //! Check if a given Key has a function that can be used without any
  //! conversions.
  /*!
   * Given a conversion map, available execution contexts, input formats, and a
   * key, determines whether the key has a corresponding function and that the
   * available contexts allow that function to be executed. \param map the map
   * between Keys and Functions used to find the needed function \param key the
   * Key representing the input formats. \param packed_formats a vector of the
   * input Format* needed for conversion. \param contexts Contexts available for
   * execution of the preprocessing \return true if the key has a matching
   * function that can be used with the inputs without any conversions.
   */
  bool CheckIfKeyMatches(ConversionMap map, Key key,
                         std::vector<format::Format *> packed_formats,
                         std::vector<context::Context *> contexts);
  //! A variadic method to pack objects into a vector
  template <typename Object, typename... Objects>
  std::vector<Object> PackObjects(Object object, Objects... objects);
  //! Base case of a variadic method to pack objects into a vector
  template <typename Object>
  std::vector<Object> PackObjects(Object object);
  //! Executes preprocessing on input formats (given variadically)
  /*!
   * Determines the function needed to carry out preprocessing on input Format*
   * objects (given variadically), as well as the Format conversions needed on
   * the inputs, executes the preprocessing, and returns the results. Note: this
   * function will delete any intermediery Format objects that were created due
   * to a conversion.
   * \param params a polymorphic pointer at the
   * object containing hyperparameters needed for preprocessing.
   * \param contexts Contexts available for execution of the
   * preprocessing.
   * \param convert_input whether or not to convert the input formats if that is
   * needed.
   * \param sf a single input Format* (this is templated to allow
   * variadic definition).
   * \param sfs a variadic Format* (this is templated to
   * allow variadic definition).
   * \return the output of the preprocessing (of
   * type ReturnType).
   */
  template <typename F, typename... SF>
  ReturnType Execute(utils::Parameters *params,
                     std::vector<context::Context *> contexts,
                     bool convert_input, F sf, SF... sfs);
  //! Executes preprocessing on input formats (given variadically)
  /*!
   * Determines the function needed to carry out preprocessing on input Format*
   * objects (given variadically), as well as the Format conversions needed on
   * the inputs, executes the preprocessing, and returns:
   * - the preprocessing result.
   * - pointers at any Format objects that were created due to a conversion.
   * Note: this function will delete any intermediery Format objects that were
   * created due to a conversion.
   * \param PreprocessParams a polymorphic pointer
   * at the object containing hyperparameters needed for preprocessing.
   * \param contexts Contexts available for execution of the
   * preprocessing.
   * \param convert_input whether or not to convert the input formats if that is
   * needed
   * \param sf a single input Format* (this is templated to allow
   * variadic definition).
   * \param sfs a variadic Format* (this is templated to
   * allow variadic definition).
   * \return a tuple containing a) the output of the
   * preprocessing (of type ReturnType), and b) a vector of Format*, where each
   * pointer in the output points at the format that the corresponds Format
   * object from the the input was converted to. If an input Format wasn't
   * converted, the output pointer will point at nullptr.
   */
  template <typename F, typename... SF>
  std::tuple<std::vector<std::vector<format::Format *>>, ReturnType>
  CachedExecute(utils::Parameters *params,
  std::vector<context::Context *> contexts, bool convert_input,
  bool clear_intermediate, F format, SF... formats);
};

template <typename ReturnType, class PreprocessingImpl, typename Function,
    typename Key, typename KeyHash, typename KeyEqualTo>
template <typename F, typename... SF>
std::tuple<std::vector<std::vector<format::Format *>>, ReturnType>
FunctionMatcherMixin<ReturnType, PreprocessingImpl, Function, Key, KeyHash,
    KeyEqualTo>::CachedExecute(utils::Parameters *params,
std::vector<context::Context *>
    contexts,
bool convert_input,
bool clear_intermediate,
    F format, SF... formats) {
ConversionMap map = this->map_to_function_;
// pack the Formats into a vector
std::vector<format::Format *> packed_formats =
    PackObjects(format, formats...);
// pack the types of Formats into a vector
std::vector<std::type_index> packed_format_types;
for (auto f : packed_formats)
packed_format_types.push_back(f->get_id());
// get conversion schema
std::tuple<Function, converter::ConversionSchema> ret =
    GetFunction(packed_formats, packed_format_types, map, contexts);
Function func = std::get<0>(ret);
converter::ConversionSchema cs = std::get<1>(ret);
// carry out conversion
// ready_formats contains the format to use in preprocessing
if (!convert_input) {
for (const auto &conversion_chain : cs) {
if (conversion_chain)
throw utils::DirectExecutionNotAvailableException(
    packed_format_types, this->GetAvailableFormats());
}
}
std::vector<std::vector<format::Format *>> all_formats =
    sparsebase::converter::Converter::ApplyConversionSchema(
        cs, packed_formats, clear_intermediate);
// The formats that will be used in the preprocessing implementation function
// calls
std::vector<format::Format *> final_formats;
std::transform(all_formats.begin(), all_formats.end(),
    std::back_inserter(final_formats),
[](std::vector<format::Format *> conversion_chain) {
return conversion_chain.back();
});
// Formats that are used to get to the final formats
std::vector<std::vector<format::Format *>> intermediate_formats;
std::transform(all_formats.begin(), all_formats.end(),
    std::back_inserter(intermediate_formats),
[](std::vector<format::Format *> conversion_chain) {
if (conversion_chain.size() > 1)
return std::vector<format::Format *>(
    conversion_chain.begin() + 1, conversion_chain.end());
return std::vector<format::Format *>();
});
// carry out the correct call
return std::make_tuple(intermediate_formats, func(final_formats, params));
}

template <typename ReturnType, class PreprocessingImpl, typename Function,
    typename Key, typename KeyHash, typename KeyEqualTo>
template <typename F, typename... SF>
ReturnType FunctionMatcherMixin<
    ReturnType, PreprocessingImpl, Function, Key, KeyHash,
    KeyEqualTo>::Execute(utils::Parameters *params,
                         std::vector<context::Context *> contexts,
bool convert_input, F sf, SF... sfs) {
auto cached_output =
    CachedExecute(params, contexts, convert_input, true, sf, sfs...);
auto converted_format_chains = std::get<0>(cached_output);
auto return_object = std::get<1>(cached_output);
for (const auto &converted_format_chain : converted_format_chains) {
for (const auto &converted_format : converted_format_chain)
delete converted_format;
}
return return_object;
}

template <typename ReturnType, class PreprocessingImpl, typename Key,
    typename KeyHash, typename KeyEqualTo, typename Function>
template <typename Object>
std::vector<Object>
FunctionMatcherMixin<ReturnType, PreprocessingImpl, Key, KeyHash, KeyEqualTo,
    Function>::PackObjects(Object object) {
  return {object};
}
template <typename ReturnType, class PreprocessingImpl, typename Key,
    typename KeyHash, typename KeyEqualTo, typename Function>
template <typename Object, typename... Objects>
std::vector<Object>
FunctionMatcherMixin<ReturnType, PreprocessingImpl, Key, KeyHash, KeyEqualTo,
    Function>::PackObjects(Object object, Objects... objects) {
  std::vector<Object> v = {object};
  std::vector<Object> remainder = PackObjects(objects...);
  for (auto i : remainder) {
    v.push_back(i);
  }
  return v;
}
template <typename ReturnType, class Preprocess, typename Function,
    typename Key, typename KeyHash, typename KeyEqualTo>
bool FunctionMatcherMixin<
    ReturnType, Preprocess, Function, Key, KeyHash,
    KeyEqualTo>::RegisterFunctionNoOverride(const Key &key_of_function,
                                            const Function &func_ptr) {
  if (map_to_function_.find(key_of_function) != map_to_function_.end()) {
    return false;  // function already exists for this Key
  } else {
    map_to_function_[key_of_function] = func_ptr;
    return true;
  }
}

template <typename ReturnType, class Preprocess, typename Function,
    typename Key, typename KeyHash, typename KeyEqualTo>
void FunctionMatcherMixin<
    ReturnType, Preprocess, Function, Key, KeyHash,
    KeyEqualTo>::RegisterFunction(const Key &key_of_function,
                                  const Function &func_ptr) {
  map_to_function_[key_of_function] = func_ptr;
}
template <typename ReturnType, class Preprocess, typename Function,
    typename Key, typename KeyHash, typename KeyEqualTo>
bool FunctionMatcherMixin<ReturnType, Preprocess, Function, Key, KeyHash,
    KeyEqualTo>::UnregisterFunction(const Key &
key_of_function) {
  if (map_to_function_.find(key_of_function) == map_to_function_.end()) {
    return false;  // function already exists for this Key
  } else {
    map_to_function_.erase(key_of_function);
    return true;
  }
}
template <typename ReturnType, class PreprocessingImpl, typename Function,
    typename Key, typename KeyHash, typename KeyEqualTo>
bool FunctionMatcherMixin<
    ReturnType, PreprocessingImpl, Function, Key, KeyHash,
    KeyEqualTo>::CheckIfKeyMatches(ConversionMap map, Key key,
                                   std::vector<format::Format *> packed_sfs,
                                   std::vector<context::Context *> contexts) {
  bool match = true;
  if (map.find(key) != map.end()) {
    for (auto sf : packed_sfs) {
      bool found_context = false;
      for (auto context : contexts) {
        if (sf->get_context()->IsEquivalent(context)) {
          found_context = true;
        }
      }
      if (!found_context) match = false;
    }
  } else {
    match = false;
  }
  return match;
}
//! Return the correct function for the operation and a conversion schema to
//! convert the input formats
/*!
 * \param key defines the types of input objects (default is vector of format
 * types) \param map the map between keys and functions  \return the
 * function to be executed and the conversion schema the conversions to carry
 * out on inputs
 */
template <typename ReturnType, class PreprocessingImpl, typename Function,
    typename Key, typename KeyHash, typename KeyEqualTo>
std::tuple<Function, converter::ConversionSchema> FunctionMatcherMixin<
    ReturnType, PreprocessingImpl, Function, Key, KeyHash,
    KeyEqualTo>::GetFunction(std::vector<format::Format *> packed_sfs, Key key,
                             ConversionMap map,
                             std::vector<context::Context *> contexts) {
  converter::ConversionSchema cs;
  Function func = nullptr;
  // When function and conversion costs are added,
  // this 'if' should be removed  -- a conversion might be
  // cheaper than direct call to matching key
  if (CheckIfKeyMatches(map, key, packed_sfs, contexts)) {
    for (auto f : key) {
      cs.push_back({});
    }
    func = map[key];
    return std::make_tuple(func, cs);
  }
  // the keys of all the available functions in preprocessing
  std::vector<Key> all_keys;
  for (const auto &key_func : map) {
    all_keys.push_back(key_func.first);
  }
  // Find all the keys that can potentially run with this input
  std::vector<std::tuple<unsigned int, converter::ConversionSchema, Key>>
      usable_keys;
  for (auto potential_key : all_keys) {
    if (potential_key.size() != key.size()) continue;
    converter::ConversionSchema temp_cs;
    bool is_usable = true;
    int conversion_cost = 0;
    for (int i = 0; i < potential_key.size(); i++) {
      if (key[i] == potential_key[i]) {
        temp_cs.push_back({});
        conversion_cost += 0;  // no conversion cost
      } else {
        auto sc = packed_sfs[i]->get_converter();
        auto conversion_chain = sc->GetConversionChain(
            key[i], packed_sfs[i]->get_context(), potential_key[i], contexts);
        if (conversion_chain) {
          temp_cs.push_back(*conversion_chain);
          conversion_cost += std::get<1>(*conversion_chain);
        } else {
          is_usable = false;
        }
      }
    }
    // At this point, we can add the cost of the function with key
    // "potential_key"
    if (is_usable) {
      int total_cost = conversion_cost;  // add function cost in the future
      usable_keys.push_back(
          std::make_tuple(total_cost, temp_cs, potential_key));
    }
  }
  if (usable_keys.empty()) {
    std::string message;
    message = "Could not find a function that matches the formats: {";
    for (auto f : packed_sfs) {
      message += f->get_name();
      message += " ";
    }
    message += "} using the contexts {";
    for (auto c : contexts) {
      message += c->get_name();
      message += " ";
    }
    message += "}";

    throw sparsebase::utils::FunctionNotFoundException(
        message);  // TODO: add a custom exception type
  }
  std::tuple<Function, converter::ConversionSchema> best_conversion;
  float cost = std::numeric_limits<float>::max();
  for (auto potential_usable_key : usable_keys) {
    if (cost > std::get<0>(potential_usable_key)) {
      cost = std::get<0>(potential_usable_key);
      cs = std::get<1>(potential_usable_key);
      func = map[std::get<2>(potential_usable_key)];
    }
  }
  return std::make_tuple(func, cs);
}
}
#endif  // SPARSEBASE_PROJECT_FUNCTIONMATCHERMIXIN_H
