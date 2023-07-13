#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>

int ReadInt(std::istream& is) {
    int l;
    is.read(reinterpret_cast<char*>(&l), sizeof(l));
    return l;
}

std::vector<int> ReadInts(std::istream& is) {
    int l = ReadInt(is);
    std::vector<int> res(l);
    is.read(reinterpret_cast<char*>(res.data()), sizeof(int) * l);
    return res;
}

class Group {
public:
    Group(std::istream& is, int batch_size, int num_queries)
    : kNumQueries(num_queries), kBatchSize(batch_size), kLen(20), is_(is) {
        Consume();
    }

    void Consume() {
        query_.clear();
        titles_.clear();
        labels_.clear();
        int size_in_bytes = ReadInt(is_);
        int num_alternatives = ReadInt(is_);
        query_ = ReadInts(is_);
        ReadInts(is_);
        for (int i = 0; i < num_alternatives; ++i) {
            labels_.push_back(ReadInt(is_));
            titles_.push_back(ReadInts(is_));
            ReadInts(is_);
        }
    }

    size_t Size() { return titles_.size(); }

    bool Remaining() { return is_.peek() != EOF; }

    void Print() {
        for (int idx: query_) {
            std::cout << idx << " ";
        }
        std::cout << "\n";
        int i = 0;
        for (const auto& title: titles_) {
            std::cout << "\t" << labels_[i] << " | ";
            for (int idx: title) {
                std::cout << idx << " ";
            }
            std::cout << "\n";
            ++i;
        }
    }

    void PutSingle(int* queries, int* group_sizes, int* titles, int* labels) {
        for (int i = 0; i < std::min((int) query_.size(), kLen); ++i) {
            queries[i] = query_[i] + 1;
        }
        *group_sizes = Size();
        int title_idx = 0;
        for (const auto& title: titles_) {
            for (int i = 0; i < std::min((int) title.size(), kLen); ++i) {
                titles[i] = title[i] + 1;
            }
            titles += kLen;
            labels[title_idx] = labels_[title_idx];
            ++title_idx;
        }
    }

    int Write(int* queries, int* group_sizes, int* titles, int* labels) {
        std::memset(queries, 0, sizeof(int) * kLen * kNumQueries);
        std::memset(group_sizes, 0, sizeof(int) * kNumQueries);
        std::memset(titles, 0, sizeof(int) * kLen * kBatchSize);
        std::memset(labels, 0, sizeof(int) * kBatchSize);

        int pairs_written = 0;
        for (int i = 0; i < kNumQueries; ++i) {
            while (kBatchSize < Size()) {
                if (!Remaining()) return i;
                Consume();
            }
            if (kBatchSize < Size() + pairs_written) return i;
            PutSingle(queries + i * kLen,
                      group_sizes + i,
                      titles + pairs_written * kLen,
                      labels + pairs_written);
            pairs_written += Size();
            if (!Remaining()) return i + 1;
            Consume();
        }
        return kNumQueries;
    }

private:
    const int kBatchSize;
    const int kNumQueries;
    const int kLen;
    std::istream& is_;
    std::vector<int> query_;
    std::vector<std::vector<int>> titles_;
    std::vector<int> labels_;
};

class QLTReader {
public:
    QLTReader(const char* qlt_path, int batch_size, int num_queries)
    : data_stream_(qlt_path), group_(data_stream_, batch_size, num_queries) {
        if (!data_stream_) {
            std::cerr << "Failed to open " << qlt_path << "\n";
        }
    }

    bool Remaining() { return data_stream_.peek() != EOF; }

    void Print() {
        std::cout << "Print" << std::endl;
        group_.Consume();
        group_.Print();
    }

    int Write(int* queries, int* group_sizes, int* titles, int* labels) {
        return group_.Write(queries, group_sizes, titles, labels);
    }

private:
    std::fstream data_stream_;
    Group group_;
};

extern "C" {
    QLTReader* QLTReader_new(const char* qlt_path, int batch_size, int num_queries) {
        QLTReader* ptr = new QLTReader(qlt_path, batch_size, num_queries);
        return ptr;
    }

    void QLTReader_delete(QLTReader* self) { delete self; }

    bool QLTReader_Remaining(QLTReader* self) { return self->Remaining(); }

    void QLTReader_Print(QLTReader* self) { self->Print(); }

    int QLTReader_Write(QLTReader* self, int* queries,
                         int* group_sizes, int* titles, int* labels) {
        return self->Write(queries, group_sizes, titles, labels);
    }
}
