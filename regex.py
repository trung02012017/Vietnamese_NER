import re
from vncorenlp import VnCoreNLP


"""
Regex class is responsible for creating regex features for training NER model
"""
class Regex:
    def __init__(self):
        self.detect_url = re.compile('(https|http|ftp|ssh)://[^\s\[\]\(\)\{\}]+', re.I)
        self.detect_url2 = re.compile('[^\s\[\]\(\)\{\}]+(\.com|\.net|\.vn|\.org|\.info|\.biz|\.mobi|\.tv|\.ws|\.name|\.us|\.ca|\.uk)', re.I)
        self.detect_email = re.compile('[^@|\s]+@[^@|\s]+')
        self.detect_datetime = re.compile('\d+[\-/]\d+[\-/]*\d*')

        self.normalize_space = re.compile(' +')

        self.normalize_special_mark = re.compile('(?P<special_mark>[\.,\(\)\[\]\{\};!?:“”\"\'/])')

        self.syntactic_features = {
            '<press>': ['tờ', 'tạp_chí', 'báo', 'đài', 'thông_tấn_xã', 'trang', 'blog'],

            '<province>': ['tỉnh', 'thành_phố', 'tp', 'tp.', 'huyện', 'quận', 'xã',
                           'phường', 'thị_trấn', 'thôn', 'bản', 'làng', 'xóm', 'ấp'],

            '<communist>': ['thành_ủy', 'tỉnh_ủy', 'quận_ủy',
                            'huyện_ủy', 'xã_ủy', 'đảng_ủy'],

            '<police>': ['công_an', 'cảnh_sát'],

            '<school>': ['ĐH', 'đại_học', 'CĐ', 'cao_đẳng', 'THPT', 'THCS', 'tiểu_học'],

            '<institution>': ['trường', 'học_viện', 'viện', 'institute', 'university'],

            '<company>': ['công_ty', 'công_ty_cổ_phần', 'tập_đoàn', 'hãng', 'xí_nghiệp',
                          'nhà_máy', 'phân_xưởng'],

            '<union>': ['liên_hiệp', 'hội', 'hợp_tác_xã', 'câu_lạc_bộ', 'trung_tâm',
                        'liên_đoàn', 'tổng_liên_đoàn'],

            '<military>': ['sư_đoàn', 'lữ_đoàn', 'trung_đoàn', 'tiểu_đoàn',
                           'quân_kh', 'liên_kh', 'đại_đội', 'tiểu_đội', 'binh_đoàn'],

            '<ministry_prefix>': ['bộ', 'ủy_ban'],

            '<ministry>': ['chính_trị', 'ngoại_giao', 'quốc_phòng', 'công_an', 'tư_pháp',
                           'tài_chính', 'công_thương', 'xây_dựng', 'nội_vụ', 'y_tế',
                           'ngoại_giao', 'lao_động', 'giao_thông', 'thông_tin', 'tt',
                           'giáo_dục', 'gd', 'nông_nghiệp', 'nn', 'kế_hoạch', 'kh',
                           'khoa_học', 'kh', 'văn_hóa', 'tài_nguyên', 'tn', 'dân_tộc'],

            '<department_prefix>': ['sở', 'phòng', 'ban', 'chi_cục', 'tổng_cục', 'cục'],

            '<village>': ['quận', 'q', 'q.', 'ấp', 'quán', 'kh', 'tổ',
                          'khóm', 'xóm', 'trạm', 'số', 'ngách', 'ngõ', 'thôn',
                          'xóm', 'bản', 'làng', 'phường'],

            '<region>': ['bang', 'nước', 'vùng', 'miền'],

            '<loc_prefix>': ['sông', 'núi', 'chợ', 'châ', 'đảo', 'đèo', 'cầ',
                             'đồi', 'đồn', 'thủ_đô', 'khách_sạn', 'sân_bay', 'nhà_hàng',
                             'cảng', 'đường', 'phố', 'đại_lộ', 'chung_cư', 'rạch',
                             'hồ', 'kênh', 'bảo_tàng', 'cao_tốc', 'ở', 'tại'],

            '<road>': ['tỉnh_lộ', 'quốc_lộ'],

            '<party>': ['đảng', 'đoàn', 'đội'],

            # '<per_prefix>' : ['ông', 'bà', 'anh', 'chị', 'cô', 'gì', 'chú',
            #                    'bác', 'cậ', 'mợ', 'ngài', 'giám_đốc', 'thủ_tướng',
            #                    'tổng_thống'],

        }

    def map_word_label(self, word):
        """
        Detect numbers and punctuation given a word
        :param word: word to detect
        :return: '<number>' if word is a number, <punct> if word is a punctuation and exact same word otherwise
        """
        if any(char.isdigit() for char in word):
            word = '<number>'
        elif word in [',', '<', '.', '>', '/', '?', '..', '...', '....', ':', ';', '"', u"'", '[', '{', ']',
                      '}', '|', '\\', '`', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '+',
                      '=', '’', '‘', '“', '”']:
            word = '<punct>'
        return word

    def normalize_string(self, content):
        """
        Remove unnecessary elements from a sentence
        :param content: sentence needed to be normalized
        :return:normalized sentence
        """
        content = self.normalize_special_mark.sub(' \g<special_mark> ', content)
        return self.normalize_space.sub(' ', content)

    def normalize_string_ex(self, content):
        """
        Remove unnecessary parts (url, email and datetime) from a sentence
        :param content: sentence needed to be normalized
        :return:normalized sentence
        """
        content = content.lower()
        new_content = self.detect_url.sub('<url>', content)
        new_content = self.detect_url2.sub('<url>', new_content)
        new_content = self.detect_email.sub('<email>', new_content)
        new_content = self.detect_datetime.sub('<datetime>', new_content)
        return new_content

    def run(self, word):
        word = word.lower()
        w = self.map_word_label(word)
        if w != '<number>' and w != '<punct>' and \
            w != '<url>' and w != '<email>' and w != '<datetime>':
            for k, v in self.syntactic_features.items():
                if word in v:
                    return k
            return '<other>'
        else:
            return w

    def run_ex(self, word):
        word = word.lower()
        w = self.normalize_string_ex(word)
        w = self.map_word_label(w)
        if w != '<number>' and w != '<punct>' and \
                w != '<url>' and w != '<email>' and w != '<datetime>':
            for k, v in self.syntactic_features.items():
                if word in v:
                    return k
            return '<other>'
        else:
            return w


if __name__ == '__main__':
    r = Regex()
    s = 'Sáng 3/9, sau khi hoàn tất thủ tục mua chiếc xe Hyundai i10 tại một đại lý bán ôtô trên đường Tam Trinh, ' \
        'quận Hai Bà Trưng (Hà Nội), vợ chồng anh Trịnh Thanh Phong ở Tây Mỗ, Nam Từ Liêm (Hà Nội) được nhân viên ' \
        'kinh doanh tư vấn có thể đăng ký xe, nộp thuế trước bạ trên mạng thay vì phải mang nhiều loại giấy tờ đi ' \
        'nộp trực tiếp.'

    annotator = VnCoreNLP(address="http://127.0.0.1", port=9000)

    # Input
    text = "Tên tôi là Trần Quang Trung. tôi ở số 47 phố Chính Kinh, Thanh Xuân, Hà Nội"

    # To perform word segmentation, POS tagging, NER and then dependency parsing
    annotated_text = annotator.annotate(text)

    # To perform word segmentation only
    word_segmented_text = annotator.tokenize(text)

    for s in annotated_text['sentences']:
        for w in s:
            print(w)