#include <iostream>
#include <exception>

using namespace std;

namespace sparsebase{
    class InvalidDataMember : public exception
    {
        string msg;
        public:
            InvalidDataMember(const string & f, const string & dm): msg(string("Format ") + f + string(" does not have ") + dm + string(" as a data member.")) {}
            virtual const char* what() const throw()
            {
                return msg.c_str();
            }
    };
}
