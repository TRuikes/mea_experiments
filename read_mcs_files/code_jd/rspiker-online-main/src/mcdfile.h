#include <fstream>

class MCDFile{
    public:
        MCDFile();
        void load();
        const std::string& getFilename() const;

        int getFilesize() const;
        int getFramesCount() const;

        uint64_t getTimeStart();
        uint64_t getTimeStop();
        uint64_t getTimePush();

    private:
        const std::string filename;
        uint64_t getFiletime(int cursor){
            std::fstream source;
            source.open(filename, std::fstream::binary | std::fstream::in);

            if(!source) return -1;
            uint64_t x;

            source.seekg(cursor);
            source.read((char*)&x,8);
            return x;
        }
};