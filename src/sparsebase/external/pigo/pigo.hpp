/**
 * PIGO: a parallel graph and matrix I/O and preprocessing library
 *
 * Release v0.5.
 *
 * Copyright (c) 2021, GT-TDAlab (Umit V. Catalyurek)
 * Copyright (c) 2021, Kasimir Gabert
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 */

#ifndef PIGO_HPP
#define PIGO_HPP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <omp.h>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace pigo {
    /** @brief Thrown by errors detected in PIGO */
    class Error : public ::std::runtime_error {
        public:
            template<class T>
            Error(T t) : ::std::runtime_error(t) { }
    };

    /** @brief Thrown when something is not implemented yet */
    class NotYetImplemented : public Error {
        public:
            template<class T>
            NotYetImplemented(T t) : Error(t) { }
    };

    /** Support for detecting vectors */
    template <typename T> struct is_vector:std::false_type{};
    template <typename... Args> struct is_vector<std::vector<Args...>>:std::true_type{};

    /** Support for detected shared pointers */
    template <typename T> struct is_sptr:std::false_type{};
    template <typename... Args> struct is_sptr<std::shared_ptr<Args...>>:std::true_type{};

    /** @brief Keep track of the state of an open file */
    typedef const char* FilePos;
    /** @brief Keep track of the state of an open writeable file */
    typedef char* WFilePos;

    /** @brief Performs actions on an opened PIGO file */
    class FileReader {
        private:
            /** Keep track of the end of the file */
            FilePos end;
        public:
            /** Keep track of the current position */
            FilePos d;

            /** @brief Initialize a new FileReader
             *
             * @param d the current position of the reader
             * @param end one beyond the last valid position
             */
            FileReader(FilePos d, FilePos end) :
                end(end), d(d) { }

            /** @brief Move the reader pass any comment lines */
            void skip_comments();

            /** @brief Read an integer from the file
            *
            * Note: this reads integers in base 10 only
            *
            * @return the newly read integer
            */
            template<typename T>
            T read_int();

            /** @brief Read a floating point value from the file
            *
            * @return the newly read floating point
            */
            template<typename T>
            T read_fp();

            /** @brief Read the sign value from an integer
             *
             * Reads out either + or -, as appropriate, from the given file
             * position. The file position is the incremented.
             *
             * @tparam T the type of the integer sign to return
             * @return a T value of either 1 or -1 as appropriate
             */
            template<typename T>
            T read_sign() {
                if (*d == '-') return (T)(-1);
                return (T)1;
            }

            /** @brief Determine if only spaces remain before the line
            *
            * Note that this does not increment the current position.
            *
            * @return true if there are only spaces, false otherwise
            */
            bool at_end_of_line();

            /** @brief Move the reader to the next non-int */
            void move_to_non_int();

            /** @brief Move the reader to the next non-floating point */
            void move_to_non_fp();

            /** @brief Move the reader to the next floating point */
            void move_to_fp();

            /** @brief Move to the first integer
             *
             * This will move to the first found integer. If it is already
             * on an integer, it will not move.
             */
            void move_to_first_int();

            /** @brief Move to the next integer
            *
            * Note: this will move through the current integer and then through
            * any non-integer character to get to the next integer
            */
            void move_to_next_int();

            /** @brief Move to the next signed integer
            *
            * Note: this will move through the current integer and then through
            * any non-integer character to get to the next integer
            *
            * These integers are signed and so can start with + or -
            */
            void move_to_next_signed_int();

            /** @brief Move to the next integer or newline
            *
            * Note: this will move through the current integer or newline
            * and then through any other character to get to the next
            * integer or newline
            */
            void move_to_next_int_or_nl();

            /** @brief Move to the end of the current line
             *
             * This moves through the line and finishes on the newline
             * character itself.
             */
            void move_to_eol();

            /** @brief Increment the file reader by a count */
            FileReader operator+(size_t s) {
                FileReader n {d, end};
                n.d = d + s;
                n.end = end;
                if (n.d > end) n.d = end;
                return n;
            }
            /** @brief Increment the file reader by a count */
            FileReader& operator+=(size_t s) {
                d += s;
                if (d > end) d = end;
                return *this;
            }

            /** @brief Set an additional, smaller end
             *
             * This will either set a new end, if it is smaller than the
             * current, or do nothing.
             *
             * @param new_end the new end to consider
             */
            void smaller_end(FileReader &rhs) {
                if (rhs.d < end) end = rhs.d;
            }

            /** @brief Return if the reader is able to read
             *
             * @return true if there are values to read, false otherwise
             */
            bool good() { return d < end; }

            /** @brief Return the remaining size to be read
             *
             * @return the number of bytes remaining
             */
            size_t size() { return end - d; }

            /** @brief Check whether the reader is at the string
             *
             * @param s the string to check against
             * @return true if the string matches, false otherwise
             */
            bool at_str(std::string s);

            /** @brief Return the current character of the reader
             *
             * @return character the reader is at
             */
            char peek() {
                if (d == end) return 0;
                return *d;
            }

            /** @brief Return whether the reader is at a newline or EOL
             *
             * @return true if the reader is at the end of a line, false
             *         otherwise
             */
            bool at_nl_or_eol() {
                if (d == end) return true;
                return (*d == '\n' || *d == '%' || *d == '#');
            }

            /** @brief Return whether the reader is at a '0' integer
             *
             * @return true if the reader is at a '0' integer
             */
            bool at_zero() {
                if (d >= end+1) return false;
                if (*d != '0') return false;
                if (*(d+1) >= '0' && *(d+1) <= '9') return false;
                return true;
            }
    };

    /** @brief Contains the supported file types */
    enum FileType {
        /** An edge list with size, typically .mtx files */
        MATRIX_MARKET,
        /** A file where each line is an edge. This is the simplest format
         * PIGO supports. Each line is `src dst` if it is unweighted, or
         * `src dst weight` if weighted. */
        EDGE_LIST,
        /** A binary format storing a PIGO COO */
        PIGO_COO_BIN,
        /** A binary format storing a PIGO CSR */
        PIGO_CSR_BIN,
        /** A binary format storing a PIGO DiGraph */
        PIGO_DIGRAPH_BIN,
        /** A file with a head and where each line contains an adjacency
         * list */
        GRAPH,
        /** A special format where PIGO will try to detect the input */
        AUTO
    };

    /** @brief The support PIGO file opening modes
     *
     * READ indicates read-only, whereas WRITE indicates read and write,
     * but on opening will create a new file (removing any old file.)
     * */
    enum OpenMode {
        /** Indicates a read-only file */
        READ,
        /** Indicates a writeable file, existing files will be removed */
        WRITE
    };

    /** @brief Manages a file opened for parallel access */
    class File {
        protected:
            /** Contains the data of the file and can be access in parallel */
            char* data_;
            /** Total size of the allocated data memory block */
            size_t size_;
            /** The current file position */
            FilePos fp_;
            /** The filename */
            std::string fn_;
        public:
            /** @brief Opens the given file
             *
             * @param fn the file name to open
             * @param mode the mode to open the file in (READ, WRITE)
             * @param max_size (only used when mode=WRITE) the maximum
             *        size to allocate for the file
             */
            File(std::string fn, OpenMode mode, size_t max_size=0);

            /** @brief Closes the open file and removes related memory */
            ~File() noexcept;

            /** @brief Copying File is unavailable */
            File(const File&) = delete;

            /** @brief Copying File is unavailable */
            File& operator=(const File&) = delete;

            /** @brief Move constructor */
            File(File &&o) {
                data_ = o.data_;
                size_ = o.size_;
            }

            /** @brief Move operator */
            File& operator=(File&& o);

            /** @brief Return the current file position object */
            FilePos fp() { return fp_; }

            /** @brief Seek to the specified offset in the file
             *
             * @param pos the position to seek to
             */
            void seek(size_t pos);

            /** @brief Read the next value from the file
             *
             * @tparam T the type of object to read
             *
             * @return the resulting object
             */
            template<class T> T read();

            /** @brief Read passed the given string
             *
             * This will ensure that the string exists in the file and
             * then read passed it. The file position will be at the first
             * character after the string when it is done. An error will
             * be throw if the string does not match.
             *
             * @param s the string to compare and read
             */
            void read(const std::string& s);

            /** @brief Write the given value to the file
             *
             * @tparam T the type of object to write
             * @param val the value to write
             *
             * @return the resulting object
             */
            template<class T> void write(T val);

            /** @brief Write a binary region in parallel
            *
            * @param v the region of data to write
            * @param v_size the size of the region of data to write
            */
            void parallel_write(char* v, size_t v_size);

            /** @brief Read a binary region in parallel
            *
            * @param v the region of data to save to
            * @param v_size the size of the region of data to read
            */
            void parallel_read(char* v, size_t v_size);

            /** @brief Return the size of the file */
            size_t size() { return size_; }

            /** @brief Auto-detect the file type
             *
             * This will determine the file type based on a mixture of the
             * extension, the contents of the file, and investigating the
             * structure of the lines.
             *
             * Note that this is a best-guess.
             *
             * @return the determined FileType
             */
            FileType guess_file_type();

            /** @brief Return a FileReader for this file
             *
             * @return returns a new FileReader object for this file
             */
            FileReader reader() {
                return FileReader {fp_, data_+size_};
            }
    };
    /** @brief Opens a read-only file for use in PIGO */
    class ROFile : public File {
        public:
            /** @brief Opens the given file
             *
             * @param fn the file name to open
             */
            ROFile(std::string fn) : File(fn, READ) { }
    };
    /** @brief Opens a writeable file for use in PIGO */
    class WFile : public File {
        public:
            /** @brief Opens the given file
             *
             * @param fn the file name to open
             * @param max_size the size to allocate for the file
             */
            WFile(std::string fn, size_t max_size) :
                File(fn, WRITE, max_size) { }
    };

    /** @brief Read a binary value from an open file
     *
     * Reads a binary value out from the given file position. The file
     * position object is incremented to just passed the value read out.
     *
     * @tparam T the type of object to read
     * @param[in,out] fp the file position object of the open file
     *
     * @return the value read out
     */
    template<class T> T read(FilePos &fp);

    /** @brief Write a binary value from an open file
     *
     * Writes a binary value into a file at the given position.
     * The file position object is incremented appropriately.
     *
     * @tparam T the type of object to write
     * @param[in,out] fp the file position object of the open file
     * @param val the value to write to the file
     */
    template<class T> void write(FilePos &fp, T val);

    /** @brief Return the size taken to write the given object
     *
     * @tparam T the type of the object
     * @param obj the object to return the output size
     *
     * @return size_t number of bytes used to write the object
     */
    template<typename T,
        typename std::enable_if<!std::is_integral<T>::value, bool>::type = false,
        typename std::enable_if<!std::is_floating_point<T>::value, bool>::type = false
        > inline size_t write_size(T obj);
    template<typename T,
        typename std::enable_if<std::is_integral<T>::value, bool>::type = true,
        typename std::enable_if<!std::is_floating_point<T>::value, bool>::type = false
        > inline size_t write_size(T obj);
    template<typename T,
        typename std::enable_if<!std::is_integral<T>::value, bool>::type = false,
        typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true
        > inline size_t write_size(T obj);

    /** @brief Write an ASCII value to an open file
     *
     * @tparam T the type of object to write
     * @param[in,out] fp the file position of the open file
     * @param obj the object to write
     */
    template<typename T,
        typename std::enable_if<!std::is_integral<T>::value, bool>::type = false,
        typename std::enable_if<!std::is_floating_point<T>::value, bool>::type = false
        > inline void write_ascii(FilePos &fp, T obj);
    template<typename T,
        typename std::enable_if<std::is_integral<T>::value, bool>::type = true,
        typename std::enable_if<!std::is_floating_point<T>::value, bool>::type = false
        > inline void write_ascii(FilePos &fp, T obj);
    template<typename T,
        typename std::enable_if<!std::is_integral<T>::value, bool>::type = false,
        typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true
        > inline void write_ascii(FilePos &fp, T obj);

    namespace detail {

        /** A holder for allocation implementations */
        template<bool do_alloc, typename T, bool ptr_flag, bool vec_flag, bool sptr_flag>
        struct allocate_impl_ {
            static void op_(T&, size_t) {
                throw Error("Invalid allocation strategy");
            }
        };

        /** The implementation that will not allocate */
        template<typename T, bool ptr_flag, bool vec_flag, bool sptr_flag>
        struct allocate_impl_<false, T, ptr_flag, vec_flag, sptr_flag> {
            static void op_(T&, size_t) { }
        };

        /** The raw pointer allocation implementation */
        template<typename T>
        struct allocate_impl_<true, T, true, false, false> {
            static void op_(T& it, size_t nmemb) {
                it = static_cast<T>(malloc(sizeof(*(T){nullptr})*nmemb));
                if (it == NULL)
                    throw Error("Unable to allocate");
            }
        };

        /** The vector allocation implementation */
        template<typename T>
        struct allocate_impl_<true, T, false, true, false> {
            static void op_(T& it, size_t nmemb) {
                it.resize(nmemb);
            }
        };

        /** The shared_ptr allocation implementation */
        template<typename T>
        struct allocate_impl_<true, T, false, false, true> {
            static void op_(T& it, size_t nmemb) {
                it = std::shared_ptr<typename T::element_type>(
                        new typename T::element_type[nmemb],
                        std::default_delete<typename T::element_type []>()
                        );
            }
        };

        /** @brief Allocates the given item appropriately
        *
        * @tparam T the storage type
        * @tparam do_alloc whether to allocate or not
        * @param[out] it the item that will be allocated
        * @param nmemb the number of members to allocate
        */
        template<class T, bool do_alloc=true>
        inline
        void allocate_mem_(T& it, size_t nmemb) {
            // Use the appropriate allocation strategy
            allocate_impl_<do_alloc, T,
                std::is_pointer<T>::value,
                is_vector<T>::value,
                is_sptr<T>::value
            >::op_(it, nmemb);
        }

        /** A holder for freeing implementations */
        template<bool do_free, typename T, bool ptr_flag, bool vec_flag, bool sptr_flag>
        struct free_impl_ {
            static void op_(T&) { };
        };

        /** The raw pointer allocation implementation */
        template<typename T>
        struct free_impl_<true, T, true, false, false> {
            static void op_(T& it) {
                free(it);
            }
        };

        /** The vector allocation implementation */
        template<typename T>
        struct free_impl_<true, T, false, true, false> {
            static void op_(T&) { }
        };

        /** The shared_ptr allocation implementation */
        template<typename T>
        struct free_impl_<true, T, false, false, true> {
            static void op_(T&) { }
        };

        /** @brief Frees the allocated item if the template parameter is true
        *
        * @tparam T the storage type
        * @tparam do_free whether to free the storage item or not
        * @param[out] it the item that will be allocated
        */
        template<class T, bool do_free=true>
        inline
        void free_mem_(T& it) {
            // Use the appropriate allocation strategy
            free_impl_<do_free, T,
                std::is_pointer<T>::value,
                is_vector<T>::value,
                is_sptr<T>::value
            >::op_(it);
        }

        /** The raw data retrieval implementation */
        template<typename T, bool ptr_flag, bool vec_flag, bool sptr_flag>
        struct get_raw_data_impl_;

        /** The raw pointer implementation */
        template<typename T>
        struct get_raw_data_impl_<T, true, false, false> {
            static char* op_(T& v) { return (char*)(v); }
        };

        /** The vector implementation */
        template<typename T>
        struct get_raw_data_impl_<T, false, true, false> {
            static char* op_(T& v) { return (char*)(v.data()); }
        };

        /** The shared_ptr implementation */
        template<typename T>
        struct get_raw_data_impl_<T, false, false, true> {
            static char* op_(T& v) { return (char*)(v.get()); }
        };

        /** @brief Returns a pointer to the raw data in an allocation
        *
        * @tparam T the storage type
        * @param v the storage item to get the raw data pointer of
        * @return a char pointer to the raw data
        */
        template<class T>
        inline
        char* get_raw_data_(T& v) {
            return get_raw_data_impl_<T,
                std::is_pointer<T>::value,
                is_vector<T>::value,
                is_sptr<T>::value
            >::op_(v);
        }

        /** A holder for the storage location setting implementation */
        template<class S, class T, bool ptr_flag, bool vec_flag, bool sptr_flag>
        struct set_value_impl_;

        /** The raw pointer value setting implementation */
        template<class S, class T>
        struct set_value_impl_<S, T, true, false, false> {
            static void op_(S &space, size_t offset, T val) {
                space[offset] = val;
            }
        };

        /** The vector value setting implementation */
        template<class S, class T>
        struct set_value_impl_<S, T, false, true, false> {
            static void op_(S &space, size_t offset, T val) {
                space[offset] = val;
            }
        };

        /** The shared_ptr value setting implementation */
        template<class S, class T>
        struct set_value_impl_<S, T, false, false, true> {
            static void op_(S &space, size_t offset, T val) {
                space.get()[offset] = val;
            }
        };

        /** @brief Set the given storage location to the specified value
        *
        * @param space the space to set a value in (raw pointer, vector,
        *        smart pointer)
        * @param offset the offset to set the value at
        * @param value the value to set
        */
        template<class S, class T>
        inline
        void set_value_(S &space, size_t offset, T val) {
            // Use the appropriate strategy based on the space type
            set_value_impl_<S, T,
                std::is_pointer<S>::value,
                is_vector<S>::value,
                is_sptr<S>::value
            >::op_(space, offset, val);
        }

        /** A holder for the getting storage implementation */
        template<class S, class T, bool ptr_flag, bool vec_flag, bool sptr_flag>
        struct get_value_impl_;

        /** The raw pointer get implementation */
        template<class S, class T>
        struct get_value_impl_<S, T, true, false, false> {
            static T op_(S &space, size_t offset) {
                return space[offset];
            }
        };

        /** The vector get implementation */
        template<class S, class T>
        struct get_value_impl_<S, T, false, true, false> {
            static T op_(S &space, size_t offset) {
                return space[offset];
            }
        };

        /** The shared_ptr get implementation */
        template<class S, class T>
        struct get_value_impl_<S, T, false, false, true> {
            static T op_(S &space, size_t offset) {
                return space.get()[offset];
            }
        };

        /** @brief Set the given storage location to the specified value
        *
        * @tparam S the storage type
        * @tparam T the underlying value type
        * @param space the space to get a value in (raw pointer, vector,
        *        smart pointer)
        * @param offset the offset to get the value at
        */
        template<class S, class T>
        inline
        T get_value_(S &space, size_t offset) {
            // Use the appropriate strategy based on the space type
            return get_value_impl_<S, T,
                std::is_pointer<S>::value,
                is_vector<S>::value,
                is_sptr<S>::value
            >::op_(space, offset);
        }

        /** @brief Implementation for false template parameters */
        template <bool B>
        struct if_true_i_ {
            static bool op_() { return false; }
        };
        /** @brief Implementation for true template parameters */
        template <>
        struct if_true_i_<true> {
            static bool op_() { return true; }
        };
        /** @brief Structure that returns bool template parameters
         *
         * @tparam B the boolean template parameter
         */
        template <bool B>
        bool if_true_() {
            return if_true_i_<B>::op_();
        }

    }

    /** @brief Write a binary region in parallel
     *
     * @param fp the FilePos to begin writing at. This will be incremented
     *        passed the written block
     * @param v the region of data to write
     * @param v_size the size of the region of data to write
     */
    void parallel_write(FilePos &fp, char* v, size_t v_size);

    /** @brief Read a binary region in parallel
     *
     * @param fp the FilePos to begin writing at. This will be incremented
     *        passed the read region
     * @param v the region of data to save to
     * @param v_size the size of the region of data to read
     */
    void parallel_read(FilePos &fp, char* v, size_t v_size);


    // We include the prototype here to support converting from CSR
    template<class Label, class Ordinal, class LabelStorage, class OrdinalStorage, bool weighted, class Weight, class WeightStorage>
    class CSR;

    /** @brief Holds coordinate-addressed matrices or graphs
     *
     * A COO is a fundamental object in PIGO. It is able to read a variety
     * of input formats (e.g., matrix market) and exposes the read data as
     * an edge list.
     *
     * The edge list is stored as multiple arrays, one for the x elements
     * and one for the y elements (e.g., src and dst in graphs.) The
     * storage method is a template parameter.
     *
     * @tparam Label the label data type. This type needs to be able to
     *         support the largest value read inside of the COO. In
     *         a graph this is the largest vertex ID.
     * @tparam Ordinal the ordinal data type. This type needs to
     *         support large enough values to hold the number of entries
     *         or rows in the COO. It defaults to the same type as the
     *         label type.
     * @tparam Storage the storage type of the COO. This can either be
     *         vector (std::vector<Label>), a pointer (Label*), or
     *         a shared_ptr (std::shared_ptr<Label>).
     * @tparam symmetric Ensure that the COO is symmetric, that is ensure
     *         that for every coordinate there is a corresponding
     *         symmetric coordinate with the same value.
     *         If true, this will always be the case. If false, any
     *         symmetry will be based on whether the input is.
     * @tparam keep_upper_triangle_only Only keep values that are in the
     *         upper triangle (if the coordinate is (x,y,val), only keep
     *         it in the coo if x <= y)
     *         If this is used with symmetric, then the COO will first
     *         symmetrize and then remove any value out of the upper
     *         triangle.
     *         If this is used without symmetric, then any edges with
     *         (y > x) will not be included in the COO.
     * @tparam remove_self_loops remove any self loops.
     *         If set to true, this will detect and remove any self loops
     *         (if the coordinate is (x,y,val) and (x==y), the entry will
     *         not be included in the COO.
     * @tparam weighted if true, support and use weights
     * @tparam Weight the weight data type.
     * @tparam WeightStorage the storage type for the weights. This can be
     *         a raw pointer (Weight*), a std::vector
     *         (std::vector<Weight>), or a std::shared_ptr<Weight>.
     */
    template<
        class Label=uint32_t,
        class Ordinal=Label,
        class Storage=Label*,
        bool symmetric=false,
        bool keep_upper_triangle_only=false,
        bool remove_self_loops=false,
        bool weighted=false,
        class Weight=float,
        class WeightStorage=Weight*
    >
    class COO {
        private:
            /** The X value of the coordinate */
            Storage x_;

            /** The Y value of the coordinate */
            Storage y_;

            /** The weight values */
            WeightStorage w_;

            /** The number of labels in the matrix represented */
            Label n_;

            /** The number of rows */
            Label nrows_;

            /** The number of columns */
            Label ncols_;

            /** The number of entries or non-zeros in the COO */
            Ordinal m_;

            /** @brief Reads the given file and type into the COO
             *
             * @param f the File to read
             * @param ft the FileType of the file. If uknown, AUTO can be
             *        used.
             */
            void read_(File& f, FileType ft);

            /** @brief Reads an edge list into the COO
             *
             * This is an internal function that will read an edge list
             * specific file to load the COO.
             *
             * @param r the FileReader to read with
             */
            void read_el_(FileReader& r);

            /** @brief Reads a matrix market file into the COO
             *
             * This is an internal function that will parse the matrix
             * market header and then load the COO appropriately.
             *
             * @param r the FileReader to read with
             */
            void read_mm_(FileReader& r);

            /** @brief Reads a PIGO binary COO
             *
             * @param f the File to read
             */
            void read_bin_(File& f);

            /** @brief Allocate the COO
             *
             * Allocates the memory for the COO to fit the storage format
             * requested.
             *
             * Note that m_ must be set before this is called.
             */
            void allocate_();

            /** @brief Convert a CSR into this COO
             *
             * @tparam CL the label type of the CSR
             * @tparam CO the ordinal type of the CSR
             * @tparam LStorage the label storage of the CSR
             * @tparam OStorage the ordinal storage of the CSR
             * @tparam CW the weight type of the CSR
             * @tparam CWS the weight storage type of the CSR
             * @param csr the CSR to load from
             */
            template <class CL, class CO, class LStorage, class OStorage, class CW, class CWS>
            void convert_csr_(CSR<CL, CO, LStorage, OStorage, weighted, CW, CWS>& csr);

            /** @brief Read an entry into the appropriate coordinate
             *
             * The position, along with the file reader, will be
             * incremented after reading.
             *
             * @param[in,out] coord_pos the current position in the
             *                coordinate list, e.g., the edge number
             * @param[in,out] r the current file reader
             * @param[in,out] max_row the current maxmium row label seen.
             *                If reading a label that is larger than the
             *                max label, this value will be updated.
             * @param[in,out] max_col the current maxmium col label seen.
             *                If reading a label that is larger than the
             *                max label, this value will be updated.
             * @tparam count_only if true, will not set values and will
             *                only assist in counting by moving through
             *                what would have been read.
             */
            template <bool count_only>
            void read_coord_entry_(size_t &coord_pos, FileReader &r,
                    Label &max_row, Label &max_col);

            /** @brief Copies the COO values from the other COO */
            void copy_(const COO& other) {
                #pragma omp parallel for
                for (Ordinal pos = 0; pos < m_; ++pos) {
                    Label x_val = detail::get_value_<
                                Storage,
                                Label
                            >((Storage&)(other.x_), pos);
                    detail::set_value_(x_, pos, x_val);
                    Label y_val = detail::get_value_<
                                Storage,
                                Label
                            >((Storage&)(other.y_), pos);
                    detail::set_value_(y_, pos, y_val);
                }
                if (detail::if_true_<weighted>()) {
                    #pragma omp parallel for
                    for (Ordinal pos = 0; pos < m_; ++pos) {
                        Weight w_val = detail::get_value_<
                                    WeightStorage,
                                    Weight
                                >((WeightStorage&)(other.w_), pos);
                        detail::set_value_(w_, pos, w_val);
                    }
                }
            }
        public:
            /** @brief Initialize a COO from a file
             *
             * The file type will attempt to be determined automatically.
             *
             * @param fn the filename to open
             */
            COO(std::string fn);

            /** @brief Initialize a COO from a file with a specific type
             *
             * @param fn the filename to open
             * @param ft the FileType to use
             */
            COO(std::string fn, FileType ft);

            /** @brief Initialize a COO from an open File with a specific type
             *
             * @param f the File to use
             * @param ft the FileType to use
             */
            COO(File& f, FileType ft);

            /** @brief Initialize from a CSR
             *
             * @param csr the CSR to convert from
             */
            template<class CL, class CO, typename LabelStorage, typename OrdinalStorage, class CW, class CWS>
            COO(CSR<CL, CO, LabelStorage, OrdinalStorage, weighted, CW, CWS>& csr);

            /** @brief Initialize an empty COO */
            COO() : n_(0), nrows_(0), ncols_(0), m_(0) { }

            /** @brief Provide space for copying in existing, out-of-band data */
            COO(Label n, Label nrows, Label ncols, Ordinal m) :
                    n_(n), nrows_(nrows), ncols_(ncols), m_(m) {
                allocate_();
            }

            /** @brief Retrieve the X coordinate array
             *
             * @return the X array in the format Storage
             */
            Storage& x() { return x_; }

            /** @brief Retrieve the Y coordinate array
             *
             * @return the Y array in the format Storage
             */
            Storage& y() { return y_; }

            /** @brief Retrieve the weight array
             *
             * @return the weight array in the format WeightStorage
             */
            WeightStorage& w() { return w_; }

            /** @brief Retrieves the number of entries in the COO
             *
             * @return the count of entries
             */
            Ordinal m() const { return m_; }

            /** @brief Retrieves the number of labels the COO contains
             *
             * Note: This will include any zero entry-labels. So, the
             * count is the largest seen label+1.
             *
             * @return the number of labels
             */
            Label n() const { return n_; }

            /** @brief Update the largest label */
            void set_n(Label new_n) { n_ = new_n; }

            /** @brief Update the number of rows in the matrix */
            void set_nrows(Label new_nrows) { nrows_ = new_nrows; }

            /** @brief Update the number of cols in the matrix */
            void set_ncols(Label new_ncols) { ncols_ = new_ncols; }

            /** @brief Retrieves the number of rows in the COO
             *
             * @return the number of rows
             */
            Label nrows() const { return nrows_; }

            /** @brief Retrieves the number of columns in the COO
             *
             * @return the number of columns
             */
            Label ncols() const { return ncols_; }

            /** @brief Saves the COO to a binary PIGO file */
            void save(std::string fn);

            /** @brief Write the COO out to an ASCII file */
            void write(std::string fn);

            /** @brief Utility to free consumed memory
             *
             * As an IO library, PIGO generally leaves memory cleanup to
             * downstream applications and does not always deallocate in
             * destructors. In some cases it is helpful for PIGO to
             * cleanup directly and then this can be used.
             */
            void free() {
                if (m_ > 0) {
                    detail::free_mem_(x_);
                    detail::free_mem_(y_);
                    detail::free_mem_<WeightStorage, weighted>(w_);
                    m_ = 0;
                }
            }

            /** @brief The copy constructor for creating a new COO */
            COO(const COO& other) : n_(other.n_), nrows_(other.nrows_),
                    ncols_(other.ncols_), m_(other.m_) {
                allocate_();
                copy_(other);
            }

            /** @brief The copy assignment operator */
            COO& operator=(const COO& other) {
                if (&other != this) {
                    free();
                    n_ = other.n_;
                    nrows_ = other.nrows_;
                    ncols_ = other.ncols_;
                    m_ = other.m_;
                    allocate_();
                    copy_(other);
                }

                return *this;
            }

            /** @brief Transpose the COO, swapping x and y */
            COO& transpose() {
                std::swap(x_, y_);
                return *this;
            }

            /** The output file header for reading/writing */
            static constexpr const char* coo_file_header = "PIGO-COO-v1";
    };

    /** @brief Holds weighted coordinate-addressed matrices or graphs
     *
     * WCOO is a wrapper around COO that is able to hold weights alongside
     * the coordinates. This is used either with weighted graphs or
     * non-binary matrices.
     *
     * This differs from COO by always setting the weighted flag.
     * For template parameter deatils, see COO.
     */
    template<
        class Label=uint32_t,
        class Ordinal=Label,
        class Storage=Label*,
        class Weight=float,
        class WeightStorage=Weight*,
        bool symmetric=false,
        bool keep_upper_triangle_only=false,
        bool remove_self_loops=false
    >
    using WCOO = COO<Label, Ordinal, Storage,
        symmetric, keep_upper_triangle_only, remove_self_loops,
        true, Weight, WeightStorage>;

    /** @brief Holds a pointer based weighted COO
     *
     * For template parameters, please see COO.
     */
    template<
        class Label=uint32_t,
        class Ordinal=Label,
        class Weight=float,
        bool symmetric=false,
        bool keep_upper_triangle_only=false,
        bool remove_self_loops=false
    >
    using WCOOPtr = COO<
        Label,
        Ordinal,
        Label*,
        symmetric,
        keep_upper_triangle_only,
        remove_self_loops,
        true,
        Weight,
        Weight*>;


    /** @brief Holds compressed sparse row matrices or graphs
     *
     * This is a fundamental object in PIGO. It is used to represent
     * sparse matrices and graphs. It can be loaded directly from
     * files that are in adjacency formats, such as CHACO/METIS files.
     *
     * In many cases, this is the desired format for a graph or matrix in
     * memory. When sparse graphs or matrices are delivered in COO
     * formats, such as matrix market or edge lists, they are frequently
     * converted to CSR. This class can automatically handle such
     * conversions internally.
     *
     * @tparam Label the label data type. This type needs to be able to
     *         support the largest value read inside of the COO. In
     *         a graph this is the largest vertex ID.
     * @tparam Ordinal the ordinal data type. This type needs to
     *         support large enough values to hold the number of endpoints
     *         or rows in the COO. It defaults to the same type as the
     *         label type.
     * @tparam LabelStorage the storage type of the endpoints of the CSR.
     *         This can either be vector (std::vector<Label>),
     *         a pointer (Label*), or a shared_ptr
     *         (std::shared_ptr<Label>).
     * @tparam OrdinalStorage the storage type of the offsets of the CSR.
     *         This can either be vector (std::vector<Ordinal>),
     *         a pointer (Ordinal*), or a shared_ptr
     *         (std::shared_ptr<Ordinal>).
     * @tparam weighted if true, support and use weights
     * @tparam Weight the weight data type. This type needs to be able to
     *         support the largest value read inside of the COO. In
     *         a graph this is the largest vertex ID.
     * @tparam WeightStorage the storage type for the weights. This can be
     *         a raw pointer (Weight*), a std::vector
     *         (std::vector<Weight>), or a std::shared_ptr<Weight>.
     */
    template<
        class Label=uint32_t,
        class Ordinal=Label,
        class LabelStorage=Label*,
        class OrdinalStorage=Ordinal*,
        bool weighted=false,
        class Weight=float,
        class WeightStorage=Weight*
    >
    class CSR {
        protected:
            /** The endpoints hold the labels (e.g., representing edges) */
            LabelStorage endpoints_;

            /** The offsets into the endpoints */
            OrdinalStorage offsets_;

            /** The weight values */
            WeightStorage weights_;

        private:
            /** The number of labels */
            Label n_;

            /** The number of endpoints */
            Ordinal m_;

            /** The number of rows */
            Label nrows_;

            /** The number of columns */
            Label ncols_;

            /** @brief Read the CSR from the given file and format
             *
             * @param f the File to read from
             * @param ft the FileFormat to use to read
             */
            void read_(File& f, FileType ft);

            /** @brief Read a binary CSR from disk
             *
             * This is an internal function that will populate the CSR
             * from a binary PIGO file.
             *
             * @param f the File to read from
             */
            void read_bin_(File& f);

            /** @brief Read a GRAPH file format
             *
             * This is an internal function that will load a GRAPH file
             * to build the CSR.
             *
             * @param r the FileReader to load from
             */
            void read_graph_(FileReader& r);

            /** @brief Allocate the storage for the CSR */
            void allocate_();

            /** @brief Convert a COO into this CSR
             *
             * @tparam COOLabel the label for the COO format
             * @tparam COOOrdinal the ordinal for the COO format
             * @tparam COOStorage the storage format of the COO
             * @tparam COOsym whether the COO is symmetrized
             * @tparam COOut whether the COO only keeps the upper triangle
             * @tparam COOsl whether the COO removes self loops
             * @tparam COOW the weight type of the COO
             * @tparam COOWS the weight storage type of the COO
             * @param coo the COO to load from
             */
            template <class COOLabel, class COOOrdinal, class COOStorage,
                     bool COOsym, bool COOut, bool COOsl,
                     class COOW, class COOWS>
            void convert_coo_(COO<COOLabel, COOOrdinal, COOStorage,
                    COOsym, COOut, COOsl, weighted, COOW, COOWS>&
                    coo);
        public:
            /** @brief Initialize an empty CSR */
            CSR() : n_(0), m_(0), nrows_(0), ncols_(0) { }

            /** @brief Allocate a CSR for the given size */
            CSR(Label n, Ordinal m, Label nrows, Label ncols) :
                    n_(n), m_(m), nrows_(nrows), ncols_(ncols) {
                allocate_();
            }

            /** @brief Initialize from a COO
             *
             * This creates a CSR from an already-loaded COO.
             *
             * Note that this will densely fill in all labels, so if there
             * are many empty rows there will be unnecessary space used.
             *
             * @tparam COOLabel the label for the COO format
             * @tparam COOOrdinal the ordinal for the COO format
             * @tparam COOStorage the storage format of the COO
             * @tparam COOsym whether the COO is symmetrized
             * @tparam COOut whether the COO only keeps the upper triangle
             * @tparam COOsl whether the COO removes self loops
             * @tparam COOW the weight type of the COO
             * @tparam COOWS the weight storage type of the COO
             * @param coo the COO object to load the CSR from
             */
            template <class COOLabel, class COOOrdinal, class COOStorage,
                     bool COOsym, bool COOut, bool COOsl,
                     class COOW, class COOWS>
            CSR(COO<COOLabel, COOOrdinal, COOStorage, COOsym, COOut,
                    COOsl, weighted, COOW, COOWS>& coo);

            /** @brief Initialize from a file
             *
             * The file type will attempt to be determined automatically.
             *
             * @param fn the filename to open
             */
            CSR(std::string fn);

            /** @brief Initialize from a file with a specific type
             *
             * @param fn the filename to open
             * @param ft the FileType to use
             */
            CSR(std::string fn, FileType ft);

            /** @brief Initialize from an open file with a specific type
             *
             * @param f the open File
             * @param ft the FileType to use
             */
            CSR(File& f, FileType ft);

            /** @brief Return the endpoints
             *
             * @return the endpoints in the LabelStorage format
             */
            LabelStorage& endpoints() { return endpoints_; }

            /** @brief Return the offsets
             *
             * These contain Ordinals that show the offset for the current
             * label into the endpoints. These are not pointers directly.
             *
             * @return the offsets in the OrdinalStorage format
             */
            OrdinalStorage& offsets() { return offsets_; }

            /** @brief Return the weights, if available
             *
             * This returns the WeightStorage for the weights, if the CSR
             * is weighted.
             *
             * @return the weights in the WeightStorage format
             */
            WeightStorage& weights() { return weights_; }

            /** @brief Retrieves the number of endpoints in the CSR
             *
             * @return the count of endpoints
             */
            Ordinal m() const { return m_; }

            /** @brief Retrieves the number of labels the CSR contains
             *
             * Note that this includes labels with no endpoints.
             *
             * @return the number of labels
             */
            Label n() const { return n_; }

            /** @brief Retrieves the number of rows in the CSR
             *
             * @return the number of rows
             */
            Label nrows() const { return nrows_; }

            /** @brief Retrieves the number of columns in the CSR
             *
             * @return the number of columns
             */
            Label ncols() const { return ncols_; }

            /** @brief Sort all row adjacencies in the CSR */
            void sort();

            /** @brief Return a new CSR without duplicate entries */
            template<class nL=Label, class nO=Ordinal, class nLS=LabelStorage, class nOS=OrdinalStorage, bool nw=weighted, class nW=Weight, class nWS=WeightStorage>
            CSR<nL, nO, nLS, nOS, nw, nW, nWS> new_csr_without_dups();

            /** @brief Utility to free consumed memory
             *
             * As an IO library, PIGO generally leaves memory cleanup to
             * downstream applications and does not always deallocate in
             * destructors. In some cases it is helpful for PIGO to
             * cleanup directly and then this can be used.
             */
            void free() {
                detail::free_mem_(endpoints_);
                detail::free_mem_(offsets_);
                detail::free_mem_<WeightStorage, weighted>(weights_);
            }

            /** @brief Return the size of the binary save file
             *
             * @return size_t containing the binary file size
             */
            size_t save_size () const;

            /** @brief Save the loaded CSR as a PIGO binary file
             *
             * This saves the current CSR to disk
             *
             * @param fn the filename to save as
             */
            void save(std::string fn);

            /** @brief Save the loaded CSR as a PIGO binary file
             *
             * This saves the current CSR to an open file
             *
             * @param w the File to save to
             */
            void save(File& w);

            /** The output file header for reading/writing */
            static constexpr const char* csr_file_header = "PIGO-CSR-v2";
    };

    /** @brief Holds a pointer based weighted CSR
     *
     * For template parameters, please see CSR.
     */
    template<
        class Label=uint32_t,
        class Ordinal=Label,
        class Weight=float
    >
    using WCSRPtr = CSR<
        Label,
        Ordinal,
        Label*,
        Ordinal*,
        true,
        Weight,
        Weight*>;

    /** @brief A compressed sparse column representation
     *
     * This is a transposed CSR.
     *
     * @tparam Label the label data type. This type needs to be able to
     *         support the largest value read inside of the CSR. In
     *         a graph this is the largest vertex ID.
     * @tparam Ordinal the ordinal data type. This type needs to
     *         support large enough values to hold the number of endpoints
     *         or rows in the CSR. It defaults to the same type as the
     *         label type.
     * @tparam LabelStorage the storage type of the endpoints of the CSR.
     *         This can either be vector (std::vector<Label>),
     *         a pointer (Label*), or a shared_ptr
     *         (std::shared_ptr<Label>).
     * @tparam OrdinalStorage the storage type of the offsets of the CSR.
     *         This can either be vector (std::vector<Ordinal>),
     *         a pointer (Ordinal*), or a shared_ptr
     *         (std::shared_ptr<Ordinal>).
     * @tparam weighted if true, support and use weights
     * @tparam Weight the weight data type.
     * @tparam WeightStorage the storage type for the weights. This can be
     *         a raw pointer (Weight*), a std::vector
     *         (std::vector<Weight>), or a std::shared_ptr<Weight>.
     */
    template<
        class Label=uint32_t,
        class Ordinal=Label,
        class LabelStorage=Label*,
        class OrdinalStorage=Ordinal*,
        bool weighted=false,
        class Weight=float,
        class WeightStorage=Weight*
    >
    class CSC : public CSR<
                Label,
                Ordinal,
                LabelStorage,
                OrdinalStorage,
                weighted,
                Weight,
                WeightStorage
            > {
        public:
            using CSR<
                Label,
                Ordinal,
                LabelStorage,
                OrdinalStorage,
                weighted,
                Weight,
                WeightStorage
            >::CSR;
    };



    /** @brief Used to load symmetric matrices from disk
     *
     * This class provides a matrix-specific naming on top of CSRs.
     *
     * @tparam Label the label data type. This type needs to be able to
     *         support the largest value read inside of the Matrix.
     * @tparam Ordinal the ordinal data type. This type needs to
     *         support large enough values to hold the number of non-zeros
     *         in the Matrix. It defaults to Label.
     * @tparam LabelStorage the storage type of the endpoints of the CSR.
     *         This can either be vector (std::vector<Label>),
     *         a pointer (Label*), or a shared_ptr
     *         (std::shared_ptr<Label>).
     * @tparam OrdinalStorage the storage type of the offsets of the CSR.
     *         This can either be vector (std::vector<Ordinal>),
     *         a pointer (Ordinal*), or a shared_ptr
     *         (std::shared_ptr<Ordinal>).
     * @tparam weighted if true, support and use weights
     * @tparam Weight the weight data type.
     * @tparam WeightStorage the storage type for the weights. This can be
     *         a raw pointer (Weight*), a std::vector
     *         (std::vector<Weight>), or a std::shared_ptr<Weight>.
     */
    template<
        class Label=uint32_t,
        class Ordinal=Label,
        class LabelStorage=Label*,
        class OrdinalStorage=Ordinal*,
        bool weighted=false,
        class Weight=float,
        class WeightStorage=Weight*
    >
    class SymMatrix: public CSR<
                        Label,
                        Ordinal,
                        LabelStorage,
                        OrdinalStorage,
                        weighted,
                        Weight,
                        WeightStorage
                     > {
        public:
            using CSR<
                    Label,
                    Ordinal,
                    LabelStorage,
                    OrdinalStorage,
                    weighted,
                    Weight,
                    WeightStorage
                >::CSR;
    };

    /** @brief Used to hold regular (non-symmetric) matrices
     *
     * This contains both a CSR and a CSC
     *
     * @tparam Label the label data type. This type needs to be able to
     *         support the largest value read inside of the Matrix.
     * @tparam Ordinal the ordinal data type. This type needs to
     *         support large enough values to hold the number of non-zeros
     *         in the Matrix. It defaults to Label.
     * @tparam LabelStorage the storage type of the endpoints of the
     *         CSR/CSC.
     *         This can either be vector (std::vector<Label>),
     *         a pointer (Label*), or a shared_ptr
     *         (std::shared_ptr<Label>).
     * @tparam OrdinalStorage the storage type of the offsets of the
     *         CSR/CSC.
     *         This can either be vector (std::vector<Ordinal>),
     *         a pointer (Ordinal*), or a shared_ptr
     *         (std::shared_ptr<Ordinal>).
     * @tparam weighted if true, support and use weights
     * @tparam Weight the weight data type.
     * @tparam WeightStorage the storage type for the weights. This can be
     *         a raw pointer (Weight*), a std::vector
     *         (std::vector<Weight>), or a std::shared_ptr<Weight>.
     */
    template<
        class Label=uint32_t,
        class Ordinal=Label,
        class LabelStorage=Label*,
        class OrdinalStorage=Ordinal*,
        bool weighted=false,
        class Weight=float,
        class WeightStorage=Weight*
    >
    class Matrix {
        private:
            /** Holds the CSR, allowing for row-based access */
            CSR<
                    Label,
                    Ordinal,
                    LabelStorage,
                    OrdinalStorage,
                    weighted,
                    Weight,
                    WeightStorage
                > csr_;

            /** Holds the CSC, allowing for col-based access */
            CSC<
                    Label,
                    Ordinal,
                    LabelStorage,
                    OrdinalStorage,
                    weighted,
                    Weight,
                    WeightStorage
                > csc_;

            /** @brief Build a Matrix from a COO */
            template <class COOLabel, class COOOrdinal, class COOStorage,
                     bool COOsym, bool COOut, bool COOsl,
                     class COOW, class COOWS>
            void from_coo_(COO<COOLabel, COOOrdinal, COOStorage, COOsym, COOut,
                    COOsl, weighted, COOW, COOWS>& coo) {
                auto coo_copy = coo;
                coo_copy.transpose();

                csc_ = CSC<
                            Label,
                            Ordinal,
                            LabelStorage,
                            OrdinalStorage,
                            weighted,
                            Weight,
                            WeightStorage
                        > { coo_copy };
                coo_copy.free();

                csr_ = CSR<
                            Label,
                            Ordinal,
                            LabelStorage,
                            OrdinalStorage,
                            weighted,
                            Weight,
                            WeightStorage
                        > { coo };
            }

        public:
            /** @brief Initialize from a COO
             *
             * This creates a Matrix from an already-loaded COO.
             *
             * This first copies the COO and transposes the copy, setting
             * that as the CSC.
             *
             * @tparam COOLabel the label for the COO format
             * @tparam COOOrdinal the ordinal for the COO format
             * @tparam COOStorage the storage format of the COO
             * @tparam COOsym whether the COO is symmetrized
             * @tparam COOut whether the COO only keeps the upper triangle
             * @tparam COOsl whether the COO removes self loops
             * @tparam COOW the weight type of the COO
             * @tparam COOWS the weight storage type of the COO
             * @param coo the COO object to load the CSR from
             */
            template <class COOLabel, class COOOrdinal, class COOStorage,
                     bool COOsym, bool COOut, bool COOsl,
                     class COOW, class COOWS>
            Matrix(COO<COOLabel, COOOrdinal, COOStorage, COOsym, COOut,
                    COOsl, weighted, COOW, COOWS>& coo) {
                from_coo_(coo);
            }

            /** @brief Build a Matrix from a file
             *
             * @param filename the file to read and load
             */
            Matrix(std::string filename) {
                COO<
                    Label, Ordinal, LabelStorage,
                    false, false, false,
                    weighted, Weight, WeightStorage
                > coo {filename};
                from_coo_(coo);
                coo.free();
            }

            /** @brief Free the associated memory */
            void free() {
                csc_.free();
                csr_.free();
            }

            /** @brief Return the number of rows */
            Ordinal nrows() { return csr_.nrows(); }

            /** @brief Return the number of columns */
            Ordinal ncols() { return csr_.ncols(); }

            /** @brief Return the CSR */
            CSR<
                Label,
                Ordinal,
                LabelStorage,
                OrdinalStorage,
                weighted,
                Weight,
                WeightStorage
            >& csr() { return csr_; }

            /** @brief Return the CSC */
            CSC<
                Label,
                Ordinal,
                LabelStorage,
                OrdinalStorage,
                weighted,
                Weight,
                WeightStorage
            >& csc() { return csc_; }

    };

    /** @brief An iterator type for edges
     *
     * @tparam V the vertex ID type
     * @tparam O the ordinal type
     * @tparam S the storage type for the edges
     */
    template<class V, class O, class S>
    class EdgeItT {
        private:
            /** The current position in the edge list */
            O pos;
            /** The edge list in the given storage format */
            S s;
        public:
            /** @brief Initialize the edge list iterator type
            *
            * @param s the storage for the edge list endpoints
            * @param pos the position in the storage
            */
            EdgeItT(S s, O pos) : pos(pos), s(s) { }
            bool operator!=(EdgeItT& rhs) { return pos != rhs.pos; }
            V& operator*();
            void operator++() { ++pos; }
    };

    /** @brief An iterable edge list
     *
     * @tparam V the vertex ID type
     * @tparam O the ordinal type
     * @tparam S the storage type for the edges
     * @param begin the starting offset
     * @param end one passed the starting offset
     */
    template<class V, class O, class S>
    class EdgeIt {
        private:
            /** The beginning position */
            O begin_;
            /** The ending position */
            O end_;
            /** The storage */
            S s;
        public:
            /** @brief Construct the edge iterator for the given offset values
            *
            * @param begin the starting offset
            * @param end the ending offset
            * @param s the storage
            */
            EdgeIt(O begin, O end, S s) : begin_(begin),
                    end_(end), s(s) { }

            /** Return the iterator start */
            EdgeItT<V,O,S> begin() { return EdgeItT<V,O,S> {s, begin_}; }
            /** Return the iterator end */
            EdgeItT<V,O,S> end() { return EdgeItT<V,O,S> {s, end_}; }
    };


    /** @brief Used to load graphs from disk
     *
     * This class provides a graph-specific naming on top of CSRs.
     *
     * @tparam vertex_t the type to use for vertices
     * @tparam edge_ctr_t the type to use for an edge counter
     * @tparam edge_storage the storage type of the endpoints of the CSR.
     * @tparam edge_ctr_storage the storage type of the offsets of the
     *         CSR.
     * @tparam weighted if true, support and use weights
     * @tparam Weight the weight data type.
     * @tparam WeightStorage the storage type for the weights. This can be
     *         a raw pointer (Weight*), a std::vector
     *         (std::vector<Weight>), or a std::shared_ptr<Weight>.
     */
    template<
        class vertex_t=uint32_t,
        class edge_ctr_t=uint32_t,
        class edge_storage=vertex_t*,
        class edge_ctr_storage=edge_ctr_t*,
        bool weighted=false,
        class Weight=float,
        class WeightStorage=Weight*
    >
    class BaseGraph : public CSR<
                        vertex_t,
                        edge_ctr_t,
                        edge_storage,
                        edge_ctr_storage,
                        weighted,
                        Weight,
                        WeightStorage
                    > {
        public:
            using CSR<vertex_t, edge_ctr_t, edge_storage, edge_ctr_storage,
                  weighted, Weight, WeightStorage>::CSR;

            /** @brief Return the neighbors of a vertex
             *
             * @param v the vertex to find the neighbors at
             * @return an edge_ctr_t containing the neighbor start offset
             */
            edge_ctr_t neighbor_start(vertex_t v);

            /** @brief Return beyond the neighbors of a vertex
             *
             * @param v the vertex to end the neighbors at
             * @return an edge_ctr_t just beyond the neighbor list
             */
            edge_ctr_t neighbor_end(vertex_t v);

            /** @brief Return the number of neighbors (degree) of a vertex
             *
             * @param v the vertex to return the degree of
             * @return the number of neighbors of the vertex
             */
            edge_ctr_t degree(vertex_t v) {
                return neighbor_end(v)-neighbor_start(v);
            }

            /** @brief Return an iterator for the neighbors the vertex
             *
             * @param v the vertex to iterate over the neighbors of
             * @return an edge iterator
             */
            EdgeIt<vertex_t, edge_ctr_t, edge_storage> neighbors(vertex_t v) {
                return EdgeIt<vertex_t, edge_ctr_t, edge_storage> {neighbor_start(v),
                    neighbor_end(v), this->endpoints_};
            }
    };

    /** @brief A basic Graph suitable for most cases */
    using Graph = BaseGraph<>;

    /** @brief A Graph suitable for large graphs requiring 64-bit ids */
    using BigGraph = BaseGraph<uint64_t, uint64_t>;

    /** @brief Used to load directed graphs from disk
     *
     * This class provides a graph-specific naming on top of non-symmetric
     * matrices (Matrix).
     *
     * @tparam vertex_t the type to use for vertices
     * @tparam edge_ctr_t the type to use for an edge counter
     * @tparam edge_storage the storage type of the endpoints of the
     *         CSR/CSC.
     * @tparam edge_ctr_storage the storage type of the offsets of the
     *         CSR/CSC.
     * @tparam weighted if true, support and use weights
     * @tparam Weight the weight data type.
     * @tparam WeightStorage the storage type for the weights. This can be
     *         a raw pointer (Weight*), a std::vector
     *         (std::vector<Weight>), or a std::shared_ptr<Weight>.
     */
    template<
        class vertex_t=uint32_t,
        class edge_ctr_t=uint32_t,
        class edge_storage=vertex_t*,
        class edge_ctr_storage=edge_ctr_t*,
        bool weighted=false,
        class Weight=float,
        class WeightStorage=Weight*
    >
    class DiGraph {
        private:
            /** Holds the in-edges */
            BaseGraph<
                    vertex_t,
                    edge_ctr_t,
                    edge_storage,
                    edge_ctr_storage,
                    weighted,
                    Weight,
                    WeightStorage
                > in_;

            /** Holds the out-edges */
            BaseGraph<
                    vertex_t,
                    edge_ctr_t,
                    edge_storage,
                    edge_ctr_storage,
                    weighted,
                    Weight,
                    WeightStorage
                > out_;

            /** @brief Read the DiGraph from the given file and format
             *
             * @param f the File to read from
             * @param ft the FileFormat to use to read
             */
            void read_(File& f, FileType ft);

            /** @build a DiGraph from a COO */
            template <class COOvertex_t, class COOedge_ctr_t, class COOStorage,
                     bool COOsym, bool COOut, bool COOsl,
                     class COOW, class COOWS>
            void from_coo_(COO<COOvertex_t, COOedge_ctr_t, COOStorage, COOsym, COOut,
                    COOsl, weighted, COOW, COOWS>& coo) {
                auto coo_copy = coo;
                coo_copy.transpose();

                in_ = BaseGraph<
                            vertex_t,
                            edge_ctr_t,
                            edge_storage,
                            edge_ctr_storage,
                            weighted,
                            Weight,
                            WeightStorage
                        > { coo_copy };
                coo_copy.free();

                out_ = BaseGraph<
                            vertex_t,
                            edge_ctr_t,
                            edge_storage,
                            edge_ctr_storage,
                            weighted,
                            Weight,
                            WeightStorage
                        > { coo };
            }


        public:
            /** @brief Initialize from a COO
             *
             * This creates a DiGraph from an already-loaded COO.
             *
             * This first copies the COO and transposes the copy, setting
             * that as the in-edges.
             * The out-edges are the original COO.
             *
             * @tparam COOvertex_t the label for the COO format
             * @tparam COOedge_ctr_t the ordinal for the COO format
             * @tparam COOStorage the storage format of the COO
             * @tparam COOsym whether the COO is symmetrized
             * @tparam COOut whether the COO only keeps the upper triangle
             * @tparam COOsl whether the COO removes self loops
             * @tparam COOW the weight type of the COO
             * @tparam COOWS the weight storage type of the COO
             * @param coo the COO object to load the CSR from
             */
            template <class COOvertex_t, class COOedge_ctr_t, class COOStorage,
                     bool COOsym, bool COOut, bool COOsl,
                     class COOW, class COOWS>
            DiGraph(COO<COOvertex_t, COOedge_ctr_t, COOStorage, COOsym, COOut,
                    COOsl, weighted, COOW, COOWS>& coo) {
                from_coo_(coo);
            }

            /** @brief Initialize from a file
             *
             * The file type will attempt to be determined automatically.
             *
             * @param fn the filename to open
             */
            DiGraph(std::string fn);

            /** @brief Initialize from a file with a specific type
             *
             * @param fn the filename to open
             * @param ft the FileType to use
             */
            DiGraph(std::string fn, FileType ft);

            /** @brief Initialize from an open file with a specific type
             *
             * @param f the open File
             * @param ft the FileType to use
             */
            DiGraph(File& f, FileType ft);

            /** @brief Free the associated memory */
            void free() {
                in_.free();
                out_.free();
            }

            /** @brief Return the number of non-zeros or edges */
            edge_ctr_t m() { return out_.m(); }

            /** @brief Return the number of vertices */
            edge_ctr_t n() { return out_.n(); }

            /** @brief Return the number of rows */
            edge_ctr_t nrows() { return out_.nrows(); }

            /** @brief Return the number of columns */
            edge_ctr_t ncols() { return out_.ncols(); }

            /** @brief Return the out-edges */
            BaseGraph<
                vertex_t,
                edge_ctr_t,
                edge_storage,
                edge_ctr_storage,
                weighted,
                Weight,
                WeightStorage
            >& out() { return out_; }

            /** @brief Return the in-edges */
            BaseGraph<
                vertex_t,
                edge_ctr_t,
                edge_storage,
                edge_ctr_storage,
                weighted,
                Weight,
                WeightStorage
            >& in() { return in_; }

            /** @brief Save the loaded DiGraph as a PIGO binary file
             *
             * This saves the current DiGraph to disk
             *
             * @param fn the filename to save as
             */
            void save(std::string fn);

            /** The output file header for reading/writing */
            static constexpr const char* digraph_file_header = "PIGO-DiGraph-v1";
    };

    inline
    File::File(std::string fn, OpenMode mode, size_t max_size) :
                fn_(fn) {
        int open_mode = O_RDONLY;
        int prot = PROT_READ;
        char fopen_mode[] = "rb";
        if (mode == WRITE) {
            open_mode = O_RDWR;
            prot = PROT_WRITE | PROT_READ;
            fopen_mode[1] = '+';
            if (max_size == 0)
                throw Error("max_size is too low to write");
        } else if (max_size > 0)
            throw Error("Max_size is only used for writing");

        if (mode == WRITE) {
            // Set the file to the given size
            FILE *w_f = fopen(fn.c_str(), "w");
            if (w_f == NULL) throw Error("PIGO: Unable to open file for writing");
            if (fseeko(w_f, max_size-1, SEEK_SET) != 0) throw Error("PIGO: Seek to set size");
            if (fputc(1, w_f) != 1) throw Error("PIGO: Unable to set size");
            if (fclose(w_f) != 0) throw Error("PIGO: Unable to close new file");
        }

        // Open the file to get the total size and base the mmap on
        #ifdef __linux__
        int fd = open(fn.c_str(), open_mode | O_DIRECT);
        #else
        int fd = open(fn.c_str(), open_mode);
        #ifdef __APPLE__
        fcntl(fd, F_NOCACHE, 1);
        #endif
        #endif
        if (fd < 0) throw Error("Unable to open file");
        FILE *f = fdopen(fd, fopen_mode);
        if (f == NULL) throw Error("PIGO: fdopen file");

        // Find the file size
        if (fseeko(f, 0 , SEEK_END) != 0)  throw Error("PIGO: Unable to seek to end");
        size_ = ftello(f);
        if (size_ == (size_t)(-1)) throw Error("PIGO: Invalid size");

        if (mode == WRITE && size_ != max_size)
            throw Error("PIGO: Wrong file size of new file");

        // MMAP the space
        data_ = (char*)mmap(NULL, size_*sizeof(char), prot,
                MAP_SHARED | MAP_NORESERVE, fd, 0);
        if (data_ == MAP_FAILED) throw Error("PIGO: MMAP");
        if (fclose(f) != 0) throw Error("PIGO: Fclose");

        // Advise the mmap for performance
        if (madvise(data_, size_, MADV_WILLNEED) != 0) throw Error("PIGO: madvise");

        // Finally, set the file position
        seek(0);
    }

    inline
    File::~File() noexcept {
        if (data_) {
            munmap(data_, size_);
            data_ = nullptr;
        }
    }

    inline
    File& File::operator=(File&& o) {
        if (&o != this) {
            if (data_)
                munmap(data_, size_);
            data_ = o.data_;
            size_ = o.size_;
            o.data_ = nullptr;
        }
        return *this;
    }

    template<class T>
    inline
    T File::read() {
        return ::pigo::read<T>(fp_);
    }

    inline
    void File::read(const std::string& s) {
        FileReader r = reader();
        if (!r.at_str(s)) throw Error("Cannot read the given string");
        // Move passed it
        fp_ += s.size();
    }

    template<class T>
    inline
    void File::write(T val) {
        return ::pigo::write(fp_, val);
    }

    inline
    void File::parallel_write(char* v, size_t v_size) {
        ::pigo::parallel_write(fp_, v, v_size);
    }

    inline
    void File::parallel_read(char* v, size_t v_size) {
        ::pigo::parallel_read(fp_, v, v_size);
    }

    inline
    void File::seek(size_t pos) {
        if (pos >= size_) throw Error("seeking beyond end of file");
        fp_ = data_ + pos;
    }

    inline
    FileType File::guess_file_type() {
        // First, check for a PIGO header
        FileReader r = reader();
        if (r.at_str(COO<>::coo_file_header))
            return PIGO_COO_BIN;
        if (r.at_str(CSR<>::csr_file_header))
            return PIGO_CSR_BIN;
        if (r.at_str(DiGraph<>::digraph_file_header))
            return PIGO_DIGRAPH_BIN;
        if (r.at_str("PIGO"))
            throw Error("Unsupported PIGO binary format, likely version mismatch");
        // Check the filename for .mtx
        std::string ext_mtx { ".mtx" };
        if (fn_.size() >= ext_mtx.size() &&
                fn_.compare(fn_.size() - ext_mtx.size(), ext_mtx.size(), ext_mtx) == 0)
            return MATRIX_MARKET;

        std::string ext_g { ".graph" };
        if (fn_.size() >= ext_g.size() &&
                fn_.compare(fn_.size() - ext_g.size(), ext_g.size(), ext_g) == 0)
            return GRAPH;
        // In future version, we can add a simple CSR-like file check by
        // looking at a few lines and counting elements
        // Default to an edge list
        return EDGE_LIST;
    };

    template<class T>
    inline
    T read(FilePos &fp) {
        T res = *(T*)fp;
        fp += sizeof(T);
        return res;
    }

    template<class T>
    inline
    void write(FilePos &fp, T val) {
        *(T*)fp = val;
        fp += sizeof(T);
    }

    template<>
    inline
    void write(FilePos &fp, std::string val) {
        for (char x : val) {
            write(fp, x);
        }
    }

    namespace detail {
        template<typename T, typename std::enable_if<!std::is_signed<T>::value, bool>::type = false>
        inline
        size_t neg_size(T obj) {
            (void)obj;
            return 0;
        }

        template<typename T, typename std::enable_if<std::is_signed<T>::value, bool>::type = true>
        inline
        size_t neg_size(T obj) {
            if (obj < 0) return 1;
            return 0;
        }

        template<typename T, typename std::enable_if<!std::is_signed<T>::value, bool>::type = false>
        inline
        void write_neg_ascii(FilePos &fp, T obj) {
            (void)fp;
            (void)obj;
        }

        template<typename T, typename std::enable_if<std::is_signed<T>::value, bool>::type = true>
        inline
        void write_neg_ascii(FilePos &fp, T obj) {
            if (obj < 0) pigo::write(fp, '-');
        }

        template<typename T, typename std::enable_if<!std::is_signed<T>::value, bool>::type = false>
        inline
        T get_positive(T obj) {
            return obj;
        }
        template<typename T, typename std::enable_if<std::is_signed<T>::value, bool>::type = true>
        inline
        T get_positive(T obj) {
            if (obj < 0) return -obj;
            return obj;
        }
    }

    template<typename T,
        typename std::enable_if<!std::is_integral<T>::value, bool>::type,
        typename std::enable_if<!std::is_floating_point<T>::value, bool>::type
        >
    inline
    size_t write_size(T obj) {
        // We do not have special processing for this obj
        return std::to_string(obj).size();
    }

    template<typename T,
        typename std::enable_if<std::is_integral<T>::value, bool>::type,
        typename std::enable_if<!std::is_floating_point<T>::value, bool>::type
        >
    inline
    size_t write_size(T obj) {
        // If it is signed, and negative, it will take an additional char
        size_t res = detail::neg_size(obj);
        do {
            obj /= 10;
            ++res;
        } while (obj != 0);
        return res;
    }

    template<typename T,
        typename std::enable_if<!std::is_integral<T>::value, bool>::type,
        typename std::enable_if<std::is_floating_point<T>::value, bool>::type
        >
    inline
    size_t write_size(T obj) {
        // This can be optimized significantly by manually writing
        return std::to_string(obj).size();
    }

    template<typename T,
        typename std::enable_if<!std::is_integral<T>::value, bool>::type,
        typename std::enable_if<!std::is_floating_point<T>::value, bool>::type
        >
    inline
    void write_ascii(FilePos &fp, T obj) {
        // We do not have special processing for this obj
        write(fp, std::to_string(obj));
    }

    template<typename T,
        typename std::enable_if<std::is_integral<T>::value, bool>::type,
        typename std::enable_if<!std::is_floating_point<T>::value, bool>::type
        >
    inline
    void write_ascii(FilePos &fp, T obj) {
        detail::write_neg_ascii(fp, obj);
        obj = detail::get_positive(obj);

        size_t num_size = write_size(obj);
        size_t pos = num_size-1;
        // Write out each digit
        do {
            *((char*)fp+pos--) = (obj%10)+'0';
            obj /= 10;
        } while (obj != 0);
        fp += num_size;
    }

    template<typename T,
        typename std::enable_if<!std::is_integral<T>::value, bool>::type,
        typename std::enable_if<std::is_floating_point<T>::value, bool>::type
        >
    inline
    void write_ascii(FilePos &fp, T obj) {
        write(fp, std::to_string(obj));
    }

    inline
    void FileReader::skip_comments() {
        while (d < end && (*d == '%' || *d == '#'))
            while (d < end && (*d++ != '\n')) { }
    }

    template<typename T>
    inline
    T FileReader::read_int() {
        T res = 0;
        while (d < end && (*d < '0' || *d > '9')) ++d;

        // Read out digit by digit
        while (d < end && (*d >= '0' && *d <= '9')) {
            res = res*10 + (*d-'0');
            ++d;
        }
        return res;
    }

    template<typename T>
    inline
    T FileReader::read_fp() {
        T res = 0.0;
        while (d < end && !((*d >= '0' && *d <= '9') || *d == 'e' ||
                    *d == 'E' || *d == '-' || *d == '+' || *d == '.')) ++d;
        // Read the size
        bool positive = true;
        if (*d == '-') {
            positive = false;
            ++d;
        } else if (*d == '+') ++d;

        // Support a simple form of floating point integers
        // Note: this is not the most accurate or fastest strategy
        // (+-)AAA.BBB(eE)(+-)ZZ.YY
        // Read the 'A' part
        while (d < end && (*d >= '0' && *d <= '9')) {
            res = res*10. + (T)(*d-'0');
            ++d;
        }
        if (*d == '.') {
            ++d;
            T fraction = 0.;
            size_t fraction_count = 0;
            // Read the 'B' part
            while (d < end && (*d >= '0' && *d <= '9')) {
                fraction = fraction*10. + (T)(*d-'0');
                ++d;
                ++fraction_count;
            }
            res += fraction / std::pow(10., fraction_count);
        }
        if (*d == 'e' || *d == 'E') {
            ++d;
            T exp = read_fp<T>();
            res *= std::pow(10., exp);
        }

        if (!positive) res *= -1;
        return res;
    }

    inline
    bool FileReader::at_end_of_line() {
        FilePos td = d;
        while (td < end && *td != '\n') {
            if (*td != ' ' && *td != '\r')
                return false;
            ++td;
        }
        return true;
    }

    inline
    void FileReader::move_to_non_int() {
        while (d < end && (*d >= '0' && *d <= '9')) ++d;
    }

    inline
    void FileReader::move_to_non_fp() {
        while (d < end && ((*d >= '0' && *d <= '9') || *d == 'e' ||
                    *d == 'E' || *d == '-' || *d == '+' || *d == '.')) ++d;
    }

    inline
    void FileReader::move_to_fp() {
        while (d < end && !((*d >= '0' && *d <= '9') || *d == 'e' ||
                    *d == 'E' || *d == '-' || *d == '+' || *d == '.')) ++d;
    }

    inline
    void FileReader::move_to_first_int() {
        // Move through the non-ints and comments
        if (*d == '%' || *d == '#') skip_comments();
        while (d < end && (*d < '0' || *d > '9')) {
            ++d;
            if (*d == '%' || *d == '#') skip_comments();
        }
    }

    inline
    void FileReader::move_to_next_int() {
        // Move through the current int
        move_to_non_int();

        // Move through the non-ints to the next int
        move_to_first_int();
    }

    inline
    void FileReader::move_to_next_signed_int() {
        if (*d == '+' || *d == '-') ++d;
        move_to_non_int();

        // Move to the next integer or signed integral value
        if (*d == '%' || *d == '#') skip_comments();
        while (d < end && (*d < '0' || *d > '9') && *d != '+' && *d != '-') {
            ++d;
            if (*d == '%' || *d == '#') skip_comments();
        }
    }

    inline
    void FileReader::move_to_next_int_or_nl() {
        bool at_int = false;
        if (d < end && (*d >= '0' && *d <= '9')) at_int = true;
        // Move through the current int or newline
        move_to_non_int();
        if (d < end && *d == '\n') {
            if (at_int) return;     // We have now reached a newline
            ++d; // Move through a newline
        }

        if (*d == '%' || *d == '#') {
            skip_comments();
            --d;        // This will end at a newline
            return;
        }
        while (d < end && (*d < '0' || *d > '9') && *d != '\n') {
            ++d;
            if (*d == '%' || *d == '#') {
                skip_comments();
                --d;
                return;
            }
        }
    }

    inline
    void FileReader::move_to_eol() {
        while (d < end && *d != '\n') { ++d; }
    }

    inline
    bool FileReader::at_str(std::string s) {
        // Ensure the size is suitable for comparison
        if (d + s.size() >= end) return false;
        std::string d_str { d, d+s.size() };
        return s.compare(d_str) == 0;
    }

    inline
    void parallel_write(FilePos &fp, char* v, size_t v_size) {
        WFilePos wfp = (WFilePos)(fp);
        #pragma omp parallel
        {
            int num_threads = omp_get_num_threads();
            int thread_id = omp_get_thread_num();

            size_t my_data = v_size/num_threads;
            // Give the last thread the remaining data
            if (thread_id == num_threads-1)
                my_data = v_size-my_data*(num_threads-1);

            size_t start_pos = (thread_id*(v_size/num_threads));

            // Memcpy the region
            char* o_out = (char*)memcpy(wfp + start_pos,
                    v + start_pos, my_data);
            if (o_out != fp+start_pos)
                throw Error("Unable to write");
        }
        fp += v_size;
    }

    inline
    void parallel_read(FilePos &fp, char* v, size_t v_size) {
        #pragma omp parallel
        {
            int num_threads = omp_get_num_threads();
            int thread_id = omp_get_thread_num();

            size_t my_data = v_size/num_threads;
            // Give the last thread the remaining data
            if (thread_id == num_threads-1)
                my_data = v_size-my_data*(num_threads-1);

            size_t start_pos = (thread_id*(v_size/num_threads));

            // Memcpy the region
            char* o_in = (char*)memcpy(v + start_pos,
                    fp + start_pos, my_data);
            if (o_in != v+start_pos)
                throw Error("Unable to read");
        }
        fp += v_size;
    }

    namespace detail {
        template <bool wgt, class W, class O>
        struct weight_size_i_ {
            static size_t op_(O) { return 0; }
        };
        template <class W, class O>
        struct weight_size_i_<true, W, O> {
            static size_t op_(O m) { return sizeof(W)*m; }
        };
        template <bool wgt, class W, class O>
        size_t weight_size_(O m) {
            return weight_size_i_<wgt, W, O>::op_(m);
        }
    }

    template<class L, class O, class S, bool sym, bool ut, bool sl, bool wgt, class W, class WS>
    COO<L,O,S,sym,ut,sl,wgt,W,WS>::COO(std::string fn) : COO(fn, AUTO) { }

    template<class L, class O, class S, bool sym, bool ut, bool sl, bool wgt, class W, class WS>
    COO<L,O,S,sym,ut,sl,wgt,W,WS>::COO(std::string fn, FileType ft) {
        // Open the file for reading
        ROFile f { fn };

        read_(f, ft);
    }

    template<class L, class O, class S, bool sym, bool ut, bool sl, bool wgt, class W, class WS>
    COO<L,O,S,sym,ut,sl,wgt,W,WS>::COO(File& f, FileType ft) {
        read_(f, ft);
    }

    template<class L, class O, class S, bool sym, bool ut, bool sl, bool wgt, class W, class WS>
    void COO<L,O,S,sym,ut,sl,wgt,W,WS>::read_(File& f, FileType ft) {
        FileType ft_used = ft;
        // If the file type is AUTO, then try to detect it
        if (ft_used == AUTO) {
            ft_used = f.guess_file_type();
        }

        if (ft_used == MATRIX_MARKET) {
            FileReader r = f.reader();
            read_mm_(r);
        } else if (ft_used == EDGE_LIST) {
            FileReader r = f.reader();
            read_el_(r);
        } else if (ft_used == PIGO_COO_BIN) {
            read_bin_(f);
        } else if (ft_used == PIGO_CSR_BIN ||
                ft_used == GRAPH) {
            // First build a CSR, then convert to a COO
            CSR<L,O,S,S,wgt,W,WS> csr {f, ft_used};
            convert_csr_(csr);
            csr.free();
        } else {
            // We need to first build a CSR, then move back to a COO
            throw NotYetImplemented("Coming in v0.6");
        }
    }

    template<class L, class O, class S, bool sym, bool ut, bool sl, bool wgt, class W, class WS>
    template <class CL, class CO, class LS, class OS, class CW, class CWS>
    COO<L,O,S,sym,ut,sl,wgt,W,WS>::COO(CSR<CL,CO,LS,OS,wgt,CW,CWS>& csr) {
        convert_csr_(csr);
    }

    template<class L, class O, class S, bool sym, bool ut, bool sl, bool wgt, class W, class WS>
    template <class CL, class CO, class LS, class OS, class CW, class CWS>
    void COO<L,O,S,sym,ut,sl,wgt,W,WS>::convert_csr_(CSR<CL,CO,LS,OS,wgt,CW,CWS>& csr) {
        // First, set our sizes and allocate space
        n_ = csr.n();
        m_ = csr.m();

        allocate_();

        auto storage_offsets = csr.offsets();
        auto storage_endpoints = csr.endpoints();
        CO* offsets = (CO*)detail::get_raw_data_(storage_offsets);
        CL* endpoints = (CL*)detail::get_raw_data_(storage_endpoints);

        CW* weights = nullptr;
        if (detail::if_true_<wgt>()) {
            auto storage_weights = csr.weights();
            weights = (CW*)detail::get_raw_data_(storage_weights);
        }

        #pragma omp parallel for schedule(dynamic, 10240)
        for (L v = 0; v < n_; ++v) {
            auto start = endpoints + offsets[v];
            auto end = endpoints + offsets[v+1];
            size_t coo_cur = offsets[v];
            CW* cur_weight = nullptr;
            if (detail::if_true_<wgt>()) {
                cur_weight = weights + offsets[v];
            }
            for (auto cur = start; cur < end; ++cur, ++coo_cur) {
                detail::set_value_(x_, coo_cur, v);
                detail::set_value_(y_, coo_cur, *cur);

                if (detail::if_true_<wgt>()) {
                    detail::set_value_(w_, coo_cur, *cur_weight);
                    ++cur_weight;
                }
            }
        }
    }

    template<class L, class O, class S, bool sym, bool ut, bool sl, bool wgt, class W, class WS>
    void COO<L,O,S,sym,ut,sl,wgt,W,WS>::allocate_() {
        detail::allocate_mem_<S>(x_, m_);
        detail::allocate_mem_<S>(y_, m_);
        detail::allocate_mem_<WS,wgt>(w_, m_);
    }

    namespace detail {
        template <bool wgt, bool is_integral, bool is_signed, bool is_real, class W, class WS, bool counting>
        struct read_wgt_i_ { static inline void op_(size_t&, WS&, FileReader&) {} };

        /** Specialization for counting integral weight types */
        template <class W, class WS>
        struct read_wgt_i_<true, true, true, false, W, WS, true> {
            static inline void op_(size_t&, WS&, FileReader& r) {
                r.move_to_next_signed_int();
            }
        };

        /** Specialization for counting unsigned integral weight types */
        template <class W, class WS>
        struct read_wgt_i_<true, true, false, false, W, WS, true> {
            static inline void op_(size_t&, WS&, FileReader& r) {
                r.move_to_next_int();
            }
        };

        /** Specialization for reading integral weight types */
        template <class W, class WS>
        struct read_wgt_i_<true, true, true, false, W, WS, false> {
            static inline void op_(size_t& coord_pos, WS& ws, FileReader& r) {
                r.move_to_next_signed_int();

                W sign = r.read_sign<W>();
                W val = r.read_int<W>()*sign;
                set_value_(ws, coord_pos, val);
            }
        };

        /** Specialization for reading integral unsigned weight types */
        template <class W, class WS>
        struct read_wgt_i_<true, true, false, false, W, WS, false> {
            static inline void op_(size_t& coord_pos, WS& ws, FileReader& r) {
                r.move_to_next_int();

                W val = r.read_int<W>();
                set_value_(ws, coord_pos, val);
            }
        };

        /** Specialization for counting floating point weight types */
        template <class W, class WS>
        struct read_wgt_i_<true, false, true, true, W, WS, true> {
            static inline void op_(size_t&, WS&, FileReader& r) {
                r.move_to_fp();
                r.move_to_non_fp();
            }
        };

        /** Specialization for reading floating point weight types */
        template <class W, class WS>
        struct read_wgt_i_<true, false, true, true, W, WS, false> {
            static inline void op_(size_t& coord_pos, WS& ws, FileReader& r) {
                r.move_to_fp();
                // Read the float
                W val = r.read_fp<W>();
                set_value_(ws, coord_pos, val);
                r.move_to_non_fp();
            }
        };

        /** @brief Read or count the weight value from a FileReader
         *
         * @tparam wgt if true, will read or count appropriately
         * @tparam W the weight type
         * @tparam WS the weight storage type
         * @tparam counting if true, will count only
         * @param coord_pos the current coordinate to insert into
         * @param ws the weight storage to use
         * @param r the FileReader that is being read from
         */
        template <bool wgt, class W, class WS, bool counting>
        inline
        void read_wgt_(size_t& coord_pos, WS& ws, FileReader& r) {
            read_wgt_i_<
                wgt,
                std::is_integral<W>::value,
                std::is_signed<W>::value,
                std::is_floating_point<W>::value,
                W,
                WS,
                counting
            >::op_(coord_pos, ws, r);
        }

        template<class L, class O, class S, bool sym, bool ut, bool sl, bool wgt, class W, class WS, bool count_only>
        struct read_coord_entry_i_;

        /** Count-only implementation of reading a coord entry */
        template<class L, class O, class S, bool sym, bool ut, bool sl, bool wgt, class W, class WS>
        struct read_coord_entry_i_<L,O,S,sym,ut,sl,wgt,W,WS,true> {
            static inline void op_(S& x_, S& y_, WS& w_, size_t &coord_pos, FileReader &r, L& max_row, L& max_col) {
                L x = r.read_int<L>();
                r.move_to_next_int();
                L y = r.read_int<L>();
                read_wgt_<wgt, W, WS, true>(coord_pos, w_, r);
                if (!r.good()) return;
                r.move_to_eol();
                r.move_to_next_int();
                if (if_true_<sl>() && x == y) {
                    return read_coord_entry_i_<L,O,S,sym,ut,sl,wgt,W,WS,true>::op_(x_, y_, w_, coord_pos, r, max_row, max_col);
                }
                if (!if_true_<sym>() && if_true_<ut>() && x > y) {
                    return read_coord_entry_i_<L,O,S,sym,ut,sl,wgt,W,WS,true>::op_(x_, y_, w_, coord_pos, r, max_row, max_col);
                }
                if (if_true_<sym>() && !if_true_<ut>() && x != y) ++coord_pos;
                ++coord_pos;
            }
        };

        /** Setting implementation of reading a coord entry */
        template<class L, class O, class S, bool sym, bool ut, bool sl, bool wgt, class W, class WS>
        struct read_coord_entry_i_<L,O,S,sym,ut,sl,wgt,W,WS,false> {
            static inline void op_(S& x_, S& y_, WS& w_, size_t &coord_pos, FileReader &r, L& max_row, L& max_col) {
                L x = r.read_int<L>();
                r.move_to_next_int();
                L y = r.read_int<L>();
                read_wgt_<wgt, W, WS, false>(coord_pos, w_, r);
                if (!r.good()) return;
                r.move_to_eol();
                r.move_to_next_int();
                if (if_true_<sl>() && x == y) {
                    return read_coord_entry_i_<L,O,S,sym,ut,sl,wgt,W,WS,false>::op_(x_, y_, w_, coord_pos, r, max_row, max_col);
                }
                if (!if_true_<sym>() && if_true_<ut>() && x > y) {
                    return read_coord_entry_i_<L,O,S,sym,ut,sl,wgt,W,WS,false>::op_(x_, y_, w_, coord_pos, r, max_row, max_col);
                }
                if (if_true_<sym>() && if_true_<ut>() && x > y) std::swap(x, y);
                set_value_(x_, coord_pos, x);
                set_value_(y_, coord_pos, y);
                ++coord_pos;
                if (if_true_<sym>() && !if_true_<ut>() && x != y) {
                    if (if_true_<wgt>()) {
                        auto w = get_value_<WS, W>(w_, coord_pos-1);
                        set_value_(w_, coord_pos, w);
                    }
                    set_value_(y_, coord_pos, x);
                    set_value_(x_, coord_pos, y);
                    ++coord_pos;
                }
                if (x > max_row) max_row = x;
                if (y > max_col) max_col = y;
            }
        };

        /** Count-only implementation of reading a coord entry without
         * flags */
        template<class L, class O, class S, bool wgt, class W, class WS>
        struct read_coord_entry_i_<L,O,S,false,false,false,wgt,W,WS,true> {
            static inline void op_(S&, S&, WS&, size_t &coord_pos, FileReader &r, L&, L&) {
                r.move_to_next_int();
                if (!r.good()) return;
                r.move_to_eol();
                r.move_to_next_int();
                ++coord_pos;
            }
        };

        /** Setting implementation of reading a coord entry without flags */
        template<class L, class O, class S, bool wgt, class W, class WS>
        struct read_coord_entry_i_<L,O,S,false,false,false,wgt,W,WS,false> {
            static inline void op_(S& x_, S& y_, WS& w_, size_t &coord_pos, FileReader &r, L& max_row, L& max_col) {
                L x = r.read_int<L>();
                r.move_to_next_int();
                L y = r.read_int<L>();
                read_wgt_<wgt, W, WS, false>(coord_pos, w_, r);
                if (!r.good()) return;
                r.move_to_eol();
                r.move_to_next_int();
                set_value_(x_, coord_pos, x);
                set_value_(y_, coord_pos, y);
                ++coord_pos;
                if (x > max_row) max_row = x;
                if (y > max_col) max_col = y;
            }
        };
    }

    template<class L, class O, class S, bool sym, bool ut, bool sl, bool wgt, class W, class WS>
    template<bool count_only>
    void COO<L,O,S,sym,ut,sl,wgt,W,WS>::read_coord_entry_(size_t &coord_pos, FileReader &r,
            L& max_row, L& max_col) {
        detail::read_coord_entry_i_<L,O,S,sym,ut,sl,wgt,W,WS,count_only>::op_(x_, y_, w_, coord_pos, r, max_row, max_col);
    }

    template<class L, class O, class S, bool sym, bool ut, bool sl, bool wgt, class W, class WS>
    void COO<L,O,S,sym,ut,sl,wgt,W,WS>::read_mm_(FileReader& r) {
        // Matrix market is ready similar to edge lists, however first the
        // header is skipped
        // Furthermore, any attributes are ignored (symmetric, etc.)
        // This should change in a future PIGO version
        if (!r.at_str("%%MatrixMarket matrix coordinate"))
            throw NotYetImplemented("Unable to handle different MatrixMarket formats");

        // Read out the first line
        r.move_to_next_int();
        L nrows = r.read_int<L>()+1;        // account for MM starting at 1
        r.move_to_next_int();
        L ncols = r.read_int<L>()+1;        // account for MM starting at 1
        r.move_to_next_int();
        O nnz = r.read_int<O>();
        r.move_to_eol();
        r.move_to_next_int();

        // Now, read out the actual contents
        read_el_(r);

        // Finally, sanity check the file
        if (nrows >= nrows_)
            nrows_ = nrows;
        else {
            free();
            throw Error("Too many row labels in file contradicting header");
        }

        if (ncols >= ncols_)
            ncols_ = ncols;
        else {
            free();
            throw Error("Too many col labels in file contradicting header");
        }
        if (detail::if_true_<sym>()) {
            if (nnz > 2*m_) {
                free();
                throw Error("Header wants more non-zeros than found");
            }
        } else if (detail::if_true_<sl>()) {
            if (nnz > m_) {
                free();
                throw Error("Header wants more non-zeros than read");
            }
        } else {
            if (nnz != m_) {
                free();
                throw Error("Header contradicts number of read non-zeros");
            }
        }

        if (nrows_ > ncols_) n_ = nrows_;
        else n_ = ncols_;
    }

    template<class L, class O, class S, bool sym, bool ut, bool sl, bool wgt, class W, class WS>
    void COO<L,O,S,sym,ut,sl,wgt,W,WS>::read_el_(FileReader& r) {
        // Get the number of threads
        omp_set_dynamic(0);
        size_t num_threads = 0;
        #pragma omp parallel shared(num_threads)
        {
            #pragma omp single
            {
                num_threads = omp_get_num_threads();
            }
        }

        // This takes two passes:
        // first, count the number of newlines to determine how to
        // allocate storage
        // second, copy over the values appropriately

        std::vector<size_t> nl_offsets(num_threads);

        L max_row = 0;
        L max_col = 0;
        #pragma omp parallel reduction(max : max_row) \
                reduction(max : max_col)
        {
            size_t tid = omp_get_thread_num();

            // Find our offsets in the file
            size_t size = r.size();
            size_t tid_start_i = (tid*size)/num_threads;
            size_t tid_end_i = ((tid+1)*size)/num_threads;
            FileReader rs = r + tid_start_i;
            FileReader re = r + tid_end_i;

            // Now, move to the appropriate starting point to move off of
            // overlapping entries
            re.move_to_eol();
            re.move_to_next_int();
            if (tid != 0) {
                rs.move_to_eol();
                rs.move_to_next_int();
            } else
                rs.move_to_first_int();

            // Set our file reader to end either at the full end or at
            // the thread id local end
            rs.smaller_end(re);

            // Pass 1
            // Iterate through, counting the number of newlines
            FileReader rs_p1 = rs;
            L max_unused;
            size_t tid_nls = 0;
            while (rs_p1.good()) {
                read_coord_entry_<true>(tid_nls, rs_p1, max_unused, max_unused);
            }

            nl_offsets[tid] = tid_nls;

            // Compute a prefix sum on the newline offsets
            #pragma omp barrier
            #pragma omp single
            {
                size_t sum_nl = 0;
                for (size_t tid = 0; tid < num_threads; ++tid) {
                    sum_nl += nl_offsets[tid];
                    nl_offsets[tid] = sum_nl;
                }

                // Now, allocate the space appropriately
                m_ = nl_offsets[num_threads-1];
                allocate_();
            }
            #pragma omp barrier

            // Pass 2
            // Iterate through again, but now copying out the integers
            FileReader rs_p2 = rs;
            size_t coord_pos = 0;
            if (tid > 0)
                coord_pos = nl_offsets[tid-1];

            while (rs_p2.good()) {
                read_coord_entry_<false>(coord_pos, rs_p2, max_row, max_col);
            }
        }

        // Set the number of labels in the matrix represented by the COO
        nrows_ = max_row + 1;
        ncols_ = max_col + 1;
        if (nrows_ > ncols_) n_ = nrows_;
        else n_ = ncols_;
    }

    template<class L, class O, class S, bool sym, bool ut, bool sl, bool wgt, class W, class WS>
    void COO<L,O,S,sym,ut,sl,wgt,W,WS>::save(std::string fn) {
        // Before creating the file, we need to find the size
        size_t out_size = 0;
        std::string cfh { coo_file_header };
        out_size += cfh.size();
        // Find the template sizes
        out_size += sizeof(uint8_t)*2;
        // Find the size of the number of rows/etc.
        out_size += sizeof(L)*3+sizeof(O);
        // Finally, find the actual COO sizes
        out_size += sizeof(L)*m_*2;
        size_t w_size = detail::weight_size_<wgt, W, O>(m_);
        out_size += w_size;

        // Create the output file
        WFile w {fn, out_size};

        // Output the PIGO COO file header
        w.write(cfh);

        // Output the template sizes
        uint8_t L_size = sizeof(L);
        uint8_t O_size = sizeof(O);
        w.write(L_size);
        w.write(O_size);

        // Output the sizes and data
        w.write(nrows_);
        w.write(ncols_);
        w.write(n_);
        w.write(m_);

        // Output the data
        char* vx = detail::get_raw_data_<S>(x_);
        size_t vx_size = sizeof(L)*m_;
        w.parallel_write(vx, vx_size);

        char* vy = detail::get_raw_data_<S>(y_);
        size_t vy_size = sizeof(L)*m_;
        w.parallel_write(vy, vy_size);

        if (w_size > 0) {
            char* vw = detail::get_raw_data_<WS>(w_);
            w.parallel_write(vw, w_size);
        }
    }

    template<class L, class O, class S, bool sym, bool ut, bool sl, bool wgt, class W, class WS>
    void COO<L,O,S,sym,ut,sl,wgt,W,WS>::read_bin_(File& f) {
        // Read and confirm the header
        f.read(coo_file_header);

        // Confirm the sizes
        uint8_t L_size, O_size;
        L_size = f.read<uint8_t>();
        O_size = f.read<uint8_t>();

        if (L_size != sizeof(L)) throw Error("Invalid COO template parameters to match binary");
        if (O_size != sizeof(O)) throw Error("Invalid COO template parameters to match binary");

        // Read the sizes
        nrows_ = f.read<L>();
        ncols_ = f.read<L>();
        n_ = f.read<L>();
        m_ = f.read<O>();

        // Allocate space
        allocate_();

        // Read out the vectors
        char* vx = detail::get_raw_data_<S>(x_);
        size_t vx_size = sizeof(L)*m_;
        f.parallel_read(vx, vx_size);

        char* vy = detail::get_raw_data_<S>(y_);
        size_t vy_size = sizeof(L)*m_;
        f.parallel_read(vy, vy_size);

        size_t w_size = detail::weight_size_<wgt, W, O>(m_);
        if (w_size > 0) {
            char* vw = detail::get_raw_data_<WS>(w_);
            f.parallel_read(vw, w_size);
        }
    }

    template<class L, class O, class S, bool sym, bool ut, bool sl, bool wgt, class W, class WS>
    void COO<L,O,S,sym,ut,sl,wgt,W,WS>::write(std::string fn) {
        // Writing occurs in two passes
        // First, each thread will simulate writing and compute how the
        // space taken
        // After the first pass, the output file is allocated
        // Second, each thread actually writes

        // Get the number of threads
        omp_set_dynamic(0);
        size_t num_threads = 0;
        #pragma omp parallel shared(num_threads)
        {
            #pragma omp single
            {
                num_threads = omp_get_num_threads();
            }
        }

        std::vector<size_t> pos_offsets(num_threads+1);
        std::shared_ptr<File> f;
        #pragma omp parallel shared(f) shared(pos_offsets)
        {
            size_t tid = omp_get_thread_num();
            size_t my_size = 0;

            #pragma omp for
            for (O e = 0; e < m_; ++e) {
                auto x = detail::get_value_<S, L>(x_, e);
                my_size += write_size(x);

                // Account for the separating space
                my_size += 1;

                auto y = detail::get_value_<S, L>(y_, e);
                my_size += write_size(y);

                if (detail::if_true_<wgt>()) {
                    // Account for the separating space
                    my_size += 1;

                    auto w = detail::get_value_<WS, W>(w_, e);
                    my_size += write_size(w);
                }
                // Account for the file newline
                my_size += 1;
            }

            pos_offsets[tid+1] = my_size;
            #pragma omp barrier

            #pragma omp single
            {
                // Compute the total size and perform a prefix sum
                pos_offsets[0] = 0;
                for (size_t thread = 1; thread <= num_threads; ++thread)
                    pos_offsets[thread] = pos_offsets[thread-1] + pos_offsets[thread];

                // Allocate the file
                f = std::make_shared<File>(fn, WRITE, pos_offsets[num_threads]);
            }

            #pragma omp barrier

            FilePos my_fp = f->fp()+pos_offsets[tid];

            // Perform the second pass, actually writing out to the file
            #pragma omp for
            for (O e = 0; e < m_; ++e) {
                auto x = detail::get_value_<S, L>(x_, e);
                write_ascii(my_fp, x);
                pigo::write(my_fp, ' ');
                auto y = detail::get_value_<S, L>(y_, e);
                write_ascii(my_fp, y);
                if (detail::if_true_<wgt>()) {
                    pigo::write(my_fp, ' ');
                    auto w = detail::get_value_<WS, W>(w_, e);
                    write_ascii(my_fp, w);
                }
                pigo::write(my_fp, '\n');
            }
        }
    }

    template<class L, class O, class LS, class OS, bool wgt, class W, class WS>
    CSR<L,O,LS,OS,wgt,W,WS>::CSR(std::string fn) : CSR(fn, AUTO) { }

    template<class L, class O, class LS, class OS, bool wgt, class W, class WS>
    CSR<L,O,LS,OS,wgt,W,WS>::CSR(std::string fn, FileType ft) {
        // Open the file for reading
        ROFile f {fn};
        read_(f, ft);
    }

    template<class L, class O, class LS, class OS, bool wgt, class W, class WS>
    CSR<L,O,LS,OS,wgt,W,WS>::CSR(File& f, FileType ft) {
        read_(f, ft);
    }

    namespace detail {
        template<bool wgt>
        struct fail_if_weighted_i_ { static void op_() {} };
        template<>
        struct fail_if_weighted_i_<true> {
            static void op_() {
                throw NotYetImplemented("Not yet implemented for weights.");
            }
        };
        template<bool wgt>
        void fail_if_weighted() {
            fail_if_weighted_i_<wgt>::op_();
        }
    }

    template<class L, class O, class LS, class OS, bool wgt, class W, class WS>
    void CSR<L,O,LS,OS,wgt,W,WS>::read_(File& f, FileType ft) {
        FileType ft_used = ft;
        // If the file type is AUTO, then try to detect it
        if (ft_used == AUTO) {
            ft_used = f.guess_file_type();
        }

        if (ft_used == MATRIX_MARKET || ft_used == EDGE_LIST ||
                ft_used == PIGO_COO_BIN) {
            // First build a COO, then load here
            COO<L,O,L*, false, false, false, wgt, W, WS> coo { f, ft_used };
            convert_coo_(coo);
            coo.free();
        } else if (ft_used == PIGO_CSR_BIN) {
            read_bin_(f);
        } else if (ft_used == GRAPH) {
            detail::fail_if_weighted<wgt>();
            FileReader r = f.reader();
            read_graph_(r);
        } else
            throw NotYetImplemented("This file type is not yet supported");
    }

    template<class L, class O, class LS, class OS, bool wgt, class W, class WS>
    void CSR<L,O,LS,OS,wgt,W,WS>::allocate_() {
        detail::allocate_mem_<LS>(endpoints_, m_);
        detail::allocate_mem_<WS,wgt>(weights_, m_);
        detail::allocate_mem_<OS>(offsets_, n_+1);
    }
    namespace detail {
        template<bool wgt, class W, class WS, class COOW, class COOWS, class O>
        struct copy_weight_i_ {
            static void op_(WS&, O, COOWS&, O) {}
        };

        template<class W, class WS, class COOW, class COOWS, class O>
        struct copy_weight_i_<true, W, WS, COOW, COOWS, O> {
            static void op_(WS& w, O p, COOWS& coow, O coop) {
                COOW val = get_value_<COOWS, COOW>(coow, coop);
                set_value_<WS, W>(w, p, (W)val);
            }
        };

        template<bool wgt, class W, class WS, class COOW, class COOWS, class O>
        void copy_weight(WS& w, O offset, COOWS& coo_w, O coo_pos) {
            copy_weight_i_<wgt, W, WS, COOW, COOWS, O>::op_(w, offset,
                    coo_w, coo_pos);
        }
    }

    template<class L, class O, class LS, class OS, bool wgt, class W,
        class WS> template<class COOL, class COOO, class COOStorage, bool
            COOsym, bool COOut, bool COOsl, class COOW, class
            COOWS>
    CSR<L,O,LS,OS,wgt,W,WS>::CSR(COO<COOL,COOO,COOStorage,COOsym,
            COOut,COOsl,wgt,COOW,COOWS>
            &coo) {
        convert_coo_(coo);
    }

    template<class L, class O, class LS, class OS, bool wgt, class W,
        class WS> template<class COOL, class COOO, class COOStorage,
        bool COOsym, bool COOut, bool COOsl,
        class COOW, class COOWS>
    void CSR<L,O,LS,OS,wgt,W,WS>::convert_coo_(COO<
            COOL,COOO,COOStorage,COOsym,COOut,COOsl,wgt,COOW,COOWS>&
            coo) {
        // Set the sizes first
        n_ = coo.n();
        m_ = coo.m();
        nrows_ = coo.nrows();
        ncols_ = coo.ncols();

        // Allocate the offsets and endpoints
        allocate_();

        // This is a multi pass algorithm.
        // First, we need to count each vertex's degree and allocate the
        // space appropriately.
        // Next, we go through the degrees and change them to offsets
        // Finally, we need to go through the COO and copy memory

        // Get the number of threads
        omp_set_dynamic(0);
        size_t num_threads = 0;
        #pragma omp parallel shared(num_threads)
        {
            #pragma omp single
            {
                num_threads = omp_get_num_threads();
            }
        }

        // Temporarily keep track of the degree for each label (this can
        // easily be computed later)
        O* label_degs = new O[n_];

        // Keep track of the starting offsets for each thread
        O* start_offsets = new O[num_threads];

        // Each thread will compute the degrees for each label on its own.
        // This is then used to reduce them all
        O* all_degs = new O[n_];
        #pragma omp parallel shared(all_degs) shared(label_degs) shared(start_offsets)
        {
            size_t tid = omp_get_thread_num();

            L v_start = (tid*n_)/num_threads;
            L v_end = ((tid+1)*n_)/num_threads;
            // We need to initialize degrees to count for zero-degree
            // vertices
            #pragma omp for
            for (L v = 0; v < n_; ++v)
                all_degs[v] = 0;

            auto coo_x = coo.x();
            auto coo_y = coo.y();
            auto coo_w = coo.w();

            #pragma omp for
            for (O x_id = 0; x_id < m_; ++x_id) {
                size_t deg_inc = detail::get_value_<COOStorage, L>(coo_x, x_id);
                #pragma omp atomic
                ++all_degs[deg_inc];
            }

            // Reduce the degree vectors
            #pragma omp barrier
            // Now all degs (via all_degs) have been computed

            O my_degs = 0;
            for (L c = v_start; c < v_end; ++c) {
                O this_deg = all_degs[c];
                label_degs[c] = this_deg;
                my_degs += this_deg;
            }

            // Save our local degree count to do a prefix sum on
            start_offsets[tid] = my_degs;

            // Get a memory allocation
            // Do a prefix sum to keep everything compact by row
            #pragma omp barrier
            #pragma omp single
            {
                O total_degs = 0;
                for (size_t cur_tid = 0; cur_tid < num_threads; ++cur_tid) {
                    total_degs += start_offsets[cur_tid];
                    start_offsets[cur_tid] = total_degs;
                }
            }
            #pragma omp barrier

            // Get the starting offset
            // The prefix sum array is off by one, so the start is at zero
            O cur_offset = 0;
            if (tid > 0)
                cur_offset = start_offsets[tid-1];

            // Now, assign the offsets to each label
            for (L c = v_start; c < v_end; ++c) {
                detail::set_value_(offsets_, c, cur_offset);
                cur_offset += label_degs[c];
            }
            #pragma omp single
            {
                // Patch the last offset to the end, making for easier
                // degree computation and iteration
                detail::set_value_(offsets_, n_, m_);
            }

            #pragma omp barrier
            // Now, all offsets_ have been assigned

            // Here, we use the degrees computed earlier and treat them
            // instead as the remaining vertices, showing the current copy
            // position

            // Finally, copy over the actual endpoints
            #pragma omp for
            for (O coo_pos = 0; coo_pos < m_; ++coo_pos) {
                L src = detail::get_value_<COOStorage, L>(coo_x, coo_pos);
                L dst = detail::get_value_<COOStorage, L>(coo_y, coo_pos);

                O this_offset_pos;
                #pragma omp atomic capture
                {
                    this_offset_pos = label_degs[src];
                    label_degs[src]--;
                }
                O this_offset = detail::get_value_<OS, O>(offsets_, src+1) - this_offset_pos;
                detail::set_value_(endpoints_, this_offset, dst);
                detail::copy_weight<wgt,W,WS,COOW,COOWS>(weights_, this_offset, coo_w, coo_pos);
            }

        }

        delete [] label_degs;
        delete [] start_offsets;
        delete [] all_degs;

    }

    template<class L, class O, class LS, class OS, bool wgt, class W, class WS>
    void CSR<L,O,LS,OS,wgt,W,WS>::read_graph_(FileReader &r) {
        // Get the number of threads
        omp_set_dynamic(0);
        size_t num_threads = 0;
        #pragma omp parallel shared(num_threads)
        {
            #pragma omp single
            {
                num_threads = omp_get_num_threads();
            }
        }

        // We have a header containing the number of vertices and edges.
        // This is used to verify and check correctness
        r.move_to_first_int();
        L read_n = r.read_int<L>();
        r.move_to_next_int();
        O read_m = r.read_int<O>();
        r.move_to_eol();
        r.move_to_next_int();

        // This takes two passes:
        // first, count the number of newlines and integers to allocate
        // storage
        // second, copy over the values appropriately

        std::vector<size_t> nl_offsets(num_threads);
        std::vector<size_t> int_offsets(num_threads);
        std::vector<L> max_labels(num_threads);
        std::vector<bool> have_zeros(num_threads, false);
        bool have_zero;
        #pragma omp parallel shared(have_zero) shared(have_zeros)
        {
            size_t tid = omp_get_thread_num();

            // Find our offsets in the file
            size_t size = r.size();
            size_t tid_start_i = (tid*size)/num_threads;
            size_t tid_end_i = ((tid+1)*size)/num_threads;
            FileReader rs = r + tid_start_i;
            FileReader re = r + tid_end_i;

            // Now, move to the appropriate starting point
            re.move_to_next_int();
            if (tid != 0) {
                rs.move_to_next_int();
            } else
                rs.move_to_first_int();

            // Set our file reader to end either at the full end or at
            // the thread id local end
            rs.smaller_end(re);

            // Now, perform the first pass and count
            FileReader rs_p1 = rs;
            size_t tid_nls = 0;
            size_t tid_ints = 0;
            bool my_have_zero = false;
            while (rs_p1.good()) {
                if (rs_p1.at_nl_or_eol())
                    ++tid_nls;
                else
                    ++tid_ints;
                // Determine if this is a 0
                if (rs_p1.at_zero() && !my_have_zero)
                    my_have_zero = true;

                rs_p1.move_to_next_int_or_nl();
            }
            if (my_have_zero) {
                have_zeros[tid] = true;
            }

            #pragma omp barrier
            #pragma omp single
            {
                bool found_zero = false;
                for (size_t tid = 0; tid < num_threads; ++tid) {
                    if (have_zeros[tid]) {
                        found_zero = true;
                        break;
                    }
                }
                if (found_zero) have_zero = true;
                else have_zero = false;
            }
            #pragma omp barrier

            nl_offsets[tid] = tid_nls;
            int_offsets[tid] = tid_ints;

            // Compute a prefix sum on the offsets
            #pragma omp barrier
            #pragma omp single
            {
                size_t sum_nl = (have_zero) ? 0 : 1;
                size_t sum_ints = 0;
                for (size_t tid = 0; tid < num_threads; ++tid) {
                    sum_nl += nl_offsets[tid];
                    nl_offsets[tid] = sum_nl;

                    sum_ints += int_offsets[tid];
                    int_offsets[tid] = sum_ints;
                }

                // Now, allocate the space appropriately
                m_ = int_offsets[num_threads-1];
                n_ = nl_offsets[num_threads-1];
                nrows_ = n_;
                allocate_();
                detail::set_value_(offsets_, 0, 0);
                if (!have_zero)
                    detail::set_value_(offsets_, 1, 0);
                detail::set_value_(offsets_, n_, m_);
            }
            #pragma omp barrier

            // Pass 2: iterate through again, but now copy out the values
            // to the appropriate position in the endpoints / offsets
            L my_max = 0;
            FileReader rs_p2 = rs;
            O endpoint_pos = 0;
            L offset_pos = 0;
            if (tid > 0) {
                offset_pos = nl_offsets[tid-1];
                endpoint_pos = int_offsets[tid-1];
            } else if (!have_zero)
                offset_pos = 1;

            while (rs_p2.good()) {
                // Ignore any trailing data in the file
                if (offset_pos >= n_) break;

                // Copy and set the endpoint
                if (rs_p2.at_nl_or_eol()) {
                    // Set the offset to the current endpoint position
                    detail::set_value_(offsets_, ++offset_pos, endpoint_pos);
                    rs_p2.move_to_next_int_or_nl();
                } else {
                    // Set the actual value
                    L endpoint = rs_p2.read_int<L>();
                    if (endpoint > my_max)
                        my_max = endpoint;
                    detail::set_value_(endpoints_, endpoint_pos++, endpoint);
                    if (!rs_p2.at_nl_or_eol())
                        rs_p2.move_to_next_int_or_nl();
                }
            }
            max_labels[tid] = my_max;
        }

        if (m_ == 2*read_m) {}
        else if (m_ != read_m) throw Error("Mismatch in CSR nonzeros and header");
        if (n_ < read_n) throw Error("Mismatch in CSR newlines and header");
        else n_ = read_n;

        L col_max = max_labels[0];
        for (size_t thread = 1; thread < num_threads; ++thread) {
            if (max_labels[thread] > col_max)
                col_max = max_labels[thread];
        }
        ncols_ = col_max+1;
    }

    template<class L, class O, class LS, class OS, bool wgt, class W, class WS>
    size_t CSR<L,O,LS,OS,wgt,W,WS>::save_size() const {
        size_t out_size = 0;
        std::string cfh { csr_file_header };
        out_size += cfh.size();
        // Find the template sizes
        out_size += sizeof(uint8_t)*2;
        // Find the size of the size of the CSR
        out_size += sizeof(L)*3+sizeof(O);
        // Finally, find the actual CSR size
        size_t voff_size = sizeof(O)*(n_+1);
        size_t vend_size = sizeof(L)*m_;
        size_t w_size = detail::weight_size_<wgt, W, O>(m_);
        out_size += voff_size + vend_size + w_size;

        return out_size;
    }

    template<class L, class O, class LS, class OS, bool wgt, class W, class WS>
    void CSR<L,O,LS,OS,wgt,W,WS>::save(std::string fn) {
        // Before creating the file, we need to find the size
        size_t out_size = save_size();

        // Create the output file
        WFile w {fn, out_size};

        // Now, perform the actual save
        save(w);
    }

    template<class L, class O, class LS, class OS, bool wgt, class W, class WS>
    void CSR<L,O,LS,OS,wgt,W,WS>::save(File& w) {
        // Output the file header
        std::string cfh { csr_file_header };
        w.write(cfh);

        // Output the template sizes
        uint8_t L_size = sizeof(L);
        uint8_t O_size = sizeof(O);
        w.write(L_size);
        w.write(O_size);

        // Output the sizes and data
        w.write(n_);
        w.write(m_);
        w.write(nrows_);
        w.write(ncols_);

        size_t voff_size = sizeof(O)*(n_+1);
        size_t vend_size = sizeof(L)*m_;
        size_t w_size = detail::weight_size_<wgt, W, O>(m_);

        // Output the data
        char* voff = detail::get_raw_data_<OS>(offsets_);
        w.parallel_write(voff, voff_size);

        char* vend = detail::get_raw_data_<LS>(endpoints_);
        w.parallel_write(vend, vend_size);

        if (w_size > 0) {
            char* wend = detail::get_raw_data_<WS>(weights_);
            w.parallel_write(wend, w_size);
        }
    }

    template<class L, class O, class LS, class OS, bool wgt, class W, class WS>
    void CSR<L,O,LS,OS,wgt,W,WS>::read_bin_(File& f) {
        // Read and confirm the header
        f.read(csr_file_header);

        // Confirm the sizes
        uint8_t L_size, O_size;
        L_size = f.read<uint8_t>();
        O_size = f.read<uint8_t>();

        if (L_size != sizeof(L)) throw Error("Invalid CSR template parameters to match binary");
        if (O_size != sizeof(O)) throw Error("Invalid CSR template parameters to match binary");

        // Read the sizes
        n_ = f.read<L>();
        m_ = f.read<O>();
        nrows_ = f.read<L>();
        ncols_ = f.read<L>();

        // Allocate space
        allocate_();

        size_t voff_size = sizeof(O)*(n_+1);
        size_t vend_size = sizeof(L)*m_;

        // Read out the vectors
        char* voff = detail::get_raw_data_<OS>(offsets_);
        f.parallel_read(voff, voff_size);

        char* vend = detail::get_raw_data_<LS>(endpoints_);
        f.parallel_read(vend, vend_size);

        size_t w_size = detail::weight_size_<wgt, W, O>(m_);
        if (w_size > 0) {
            char* wend = detail::get_raw_data_<WS>(weights_);
            f.parallel_read(wend, w_size);
        }
    }

    template<class L, class O, class LS, class OS, bool wgt, class W, class WS>
    void CSR<L,O,LS,OS,wgt,W,WS>::sort() {
        #pragma omp parallel for schedule(dynamic, 10240)
        for (L v = 0; v < n_; ++v) {
            // Get the start and end range
            O start = detail::get_value_<OS, O>(offsets_, v);
            O end = detail::get_value_<OS, O>(offsets_, v+1);

            // Sort the range from the start to the end
            if (detail::if_true_<wgt>()) {
                // This should be improved, e.g., without replicating
                // everything. For now, make a joint array, sort that, and
                // then pull out the resulting data
                std::vector<std::pair<L, W>> vec;
                vec.reserve(end-start);
                for (O cur = start; cur < end; ++cur) {
                    L l = detail::get_value_<LS, L>(endpoints_, cur);
                    W w = detail::get_value_<WS, W>(weights_, cur);
                    std::pair<L, W> val = {l, w};
                    vec.emplace_back(val);
                }
                std::sort(vec.begin(), vec.end());
                O cur = start;
                for (auto& pair : vec) {
                    L l = std::get<0>(pair);
                    W w = std::get<1>(pair);
                    detail::set_value_(endpoints_, cur, l);
                    detail::set_value_(weights_, cur, w);
                    ++cur;
                }
            } else {
                L* endpoints = (L*)detail::get_raw_data_(endpoints_);

                L* range_start = endpoints+start;
                L* range_end = endpoints+end;

                std::sort(range_start, range_end);
            }
        }
    }

    template<class L, class O, class LS, class OS, bool wgt, class W, class WS>
    template<class nL, class nO, class nLS, class nOS, bool nw, class nW, class nWS>
    CSR<nL, nO, nLS, nOS, nw, nW, nWS> CSR<L,O,LS,OS,wgt,W,WS>::new_csr_without_dups() {
        // First, sort ourselves
        sort();

        // Next, count the degrees for each vertex, excluding duplicates
        std::shared_ptr<L> degs_storage;
        detail::allocate_mem_(degs_storage, n_);
        L* degs = degs_storage.get();

        nO new_m = 0;

        #pragma omp parallel for schedule(dynamic, 10240) shared(degs) reduction(+ : new_m)
        for (L v = 0; v < n_; ++v) {
            O start = detail::get_value_<OS, O>(offsets_, v);
            O end = detail::get_value_<OS, O>(offsets_, v+1);
            if (end-start == 0) {
                degs[v] = 0;
                continue;
            }

            L prev_val = detail::get_value_<LS, L>(endpoints_, start++);
            L new_deg = 1;

            while (start != end) {
                L cur_val = detail::get_value_<LS, L>(endpoints_, start++);
                if (cur_val != prev_val) {
                    prev_val = cur_val;
                    ++new_deg;
                }
            }

            degs[v] = new_deg;
            new_m += new_deg;
        }

        // Allocate the new CSR
        CSR<nL, nO, nLS, nOS, nw, nW, nWS> ret { (nL)n_, new_m, (nL)nrows_, (nL)ncols_ };

        auto& new_endpoints = ret.endpoints();
        auto& new_offsets = ret.offsets();
        auto& new_weights = ret.weights();

        // Set the offsets by doing a prefix sum on the degrees

        // Get the number of threads
        omp_set_dynamic(0);
        size_t num_threads = 0;
        #pragma omp parallel shared(num_threads)
        {
            #pragma omp single
            {
                num_threads = omp_get_num_threads();
            }
        }
        // Keep track of the starting offsets for each thread
        std::shared_ptr<O> so_storage;
        detail::allocate_mem_(so_storage, num_threads);
        O* start_offsets = so_storage.get();

        #pragma omp parallel shared(start_offsets)
        {
            size_t tid = omp_get_thread_num();

            L v_start = (tid*n_)/num_threads;
            L v_end = ((tid+1)*n_)/num_threads;

            O my_degs = 0;
            for (L c = v_start; c < v_end; ++c) {
                // FIXME get the degs
                O this_deg = degs[c];
                my_degs += this_deg;
            }

            // Save our local degree count to do a prefix sum on
            start_offsets[tid] = my_degs;

            #pragma omp barrier
            #pragma omp single
            {
                O total_degs = 0;
                for (size_t cur_tid = 0; cur_tid < num_threads; ++cur_tid) {
                    total_degs += start_offsets[cur_tid];
                    start_offsets[cur_tid] = total_degs;
                }
            }
            #pragma omp barrier

            // Get the starting offset
            // The prefix sum array is off by one, so the start is at zero
            O cur_offset = 0;
            if (tid > 0)
                cur_offset = start_offsets[tid-1];

            // Now, assign the offsets to each label
            for (L c = v_start; c < v_end; ++c) {
                detail::set_value_(new_offsets, c, cur_offset);
                cur_offset += degs[c];
            }
            #pragma omp single
            {
                // Patch the last offset to the end, making for easier
                // degree computation and iteration
                detail::set_value_(new_offsets, (nL)n_, new_m);
            }

            // Now, all new offsets have been assigned
        }

        // Repeat going through the edges, copying out the endpoints
        #pragma omp parallel for schedule(dynamic, 10240)
        for (L v = 0; v < n_; ++v) {
            O o_start = detail::get_value_<OS, O>(offsets_, v);
            O o_end = detail::get_value_<OS, O>(offsets_, v+1);
            if (o_end-o_start == 0) continue;

            nO n_cur = detail::get_value_<nOS, nO>(new_offsets, (nL)v);

            L prev_val = detail::get_value_<LS, L>(endpoints_, o_start++);
            detail::set_value_(new_endpoints, n_cur++, prev_val);
            if (detail::if_true_<nw>()) {
                W w = detail::get_value_<WS, W>(weights_, o_start-1);
                detail::set_value_(new_weights, n_cur-1, (nW)w);
            }

            while (o_start != o_end) {
                L cur_val = detail::get_value_<LS, L>(endpoints_, o_start++);
                if (prev_val != cur_val) {
                    prev_val = cur_val;
                    detail::set_value_(new_endpoints, n_cur++, (nL)prev_val);
                    if (detail::if_true_<nw>()) {
                        W w = detail::get_value_<WS, W>(weights_, o_start-1);
                        detail::set_value_(new_weights, n_cur-1, (nW)w);
                    }
                }
            }
        }

        return ret;
    }

    template<class vertex_t, class edge_ctr_t, class edge_storage, class edge_ctr_storage, bool weighted, class Weight, class WeightStorage>
    DiGraph<vertex_t, edge_ctr_t, edge_storage, edge_ctr_storage, weighted, Weight, WeightStorage>::DiGraph(std::string fn) :
        DiGraph(fn, AUTO) { }

    template<class vertex_t, class edge_ctr_t, class edge_storage, class edge_ctr_storage, bool weighted, class Weight, class WeightStorage>
    DiGraph<vertex_t, edge_ctr_t, edge_storage, edge_ctr_storage, weighted, Weight, WeightStorage>::DiGraph(std::string fn, FileType ft) {
        ROFile f {fn};
        read_(f, ft);
    }

    template<class vertex_t, class edge_ctr_t, class edge_storage, class edge_ctr_storage, bool weighted, class Weight, class WeightStorage>
    DiGraph<vertex_t, edge_ctr_t, edge_storage, edge_ctr_storage, weighted, Weight, WeightStorage>::DiGraph(File& f, FileType ft) {
        read_(f, ft);
    }

    template<class vertex_t, class edge_ctr_t, class edge_storage, class edge_ctr_storage, bool weighted, class Weight, class WeightStorage>
    void DiGraph<vertex_t, edge_ctr_t, edge_storage, edge_ctr_storage, weighted, Weight, WeightStorage>::read_(File& f, FileType ft) {
        FileType ft_used = ft;
        // If the file type is AUTO, then try to detect it
        if (ft_used == AUTO) {
            ft_used = f.guess_file_type();
        }
        if (ft_used == PIGO_DIGRAPH_BIN) {
            // First load the in, then the out
            // Read out the header
            f.read(digraph_file_header);
            in_ = BaseGraph<
                        vertex_t,
                        edge_ctr_t,
                        edge_storage,
                        edge_ctr_storage,
                        weighted,
                        Weight,
                        WeightStorage
                    > { f, AUTO };
            out_ = BaseGraph<
                        vertex_t,
                        edge_ctr_t,
                        edge_storage,
                        edge_ctr_storage,
                        weighted,
                        Weight,
                        WeightStorage
                    > { f, AUTO };
        } else {
            // Build a COO, then load ourselves from it
            COO<
                    vertex_t, edge_ctr_t, edge_storage,
                    false, false, false,
                    weighted, Weight, WeightStorage
                > coo { f, ft };
            from_coo_(coo);
            coo.free();
        }
    }

    template<class vertex_t, class edge_ctr_t, class edge_storage, class edge_ctr_storage, bool weighted, class Weight, class WeightStorage>
    void DiGraph<vertex_t, edge_ctr_t, edge_storage, edge_ctr_storage, weighted, Weight, WeightStorage>::save(std::string fn) {
        // Find the total size to save
        size_t out_size = 0;

        std::string dfh { digraph_file_header };
        out_size += dfh.size();

        out_size += in_.save_size();
        out_size += out_.save_size();

        // Now, create the file and output everything
        WFile w {fn, out_size};

        w.write(dfh);

        in_.save(w);
        out_.save(w);
    }

    template<class V, class O, class S>
    V& EdgeItT<V,O,S>::operator*() { return detail::get_value_<S, V&>(s, pos); }

    template<class vertex_t, class edge_ctr_t, class edge_storage, class edge_ctr_storage, bool weighted, class Weight, class WeightStorage>
    edge_ctr_t BaseGraph<vertex_t, edge_ctr_t, edge_storage, edge_ctr_storage, weighted, Weight, WeightStorage>::neighbor_start(vertex_t v) {
        return detail::get_value_<edge_ctr_storage, edge_ctr_t>(this->offsets_, v);
    }

    template<class vertex_t, class edge_ctr_t, class edge_storage, class edge_ctr_storage, bool weighted, class Weight, class WeightStorage>
    edge_ctr_t BaseGraph<vertex_t, edge_ctr_t, edge_storage, edge_ctr_storage, weighted, Weight, WeightStorage>::neighbor_end(vertex_t v) {
        return detail::get_value_<edge_ctr_storage, edge_ctr_t>(this->offsets_, v+1);
    }

}

#endif /* PIGO_HPP */
