# Công nghệ dùng trong bộ kiểm tra nhãn Hán Nôm

Tài liệu mô tả chi tiết phương pháp, thuật toán và dữ liệu tham chiếu của script `hannom_check.py`.

---

## 1. Quan điểm thiết kế

Bộ kiểm tra **không dùng mô hình học máy**, không dùng embedding, không gọi API mô hình ngôn ngữ. Toàn bộ là Python thuần cộng với `pandas`, chạy trên CPU, không cần GPU. Lý do: với dữ liệu nhãn (label QA), thứ cần nhất là **giải thích được** — mỗi khi máy báo một cặp chữ–âm là đáng ngờ, người rà soát phải thấy ngay lý do và dẫn chứng để quyết định. Một mô hình neural cho ra điểm số 0.37 mà không nói được vì sao thì gần như vô dụng trong quy trình sửa nhãn.

Thay vào đó, hệ thống dựng trên bốn nguồn bằng chứng độc lập:

| Nguồn bằng chứng | Bản chất | Bắt được lỗi gì |
|---|---|---|
| Ràng buộc hình thức Unicode | Tất định, đúng/sai tuyệt đối | Codepoint lệch ký tự, ký tự ẩn, ký tự ngoài khối Hán |
| Âm vị học tiếng Việt | Tất định, dựa trên cấu trúc âm tiết | Âm tiết không tồn tại trong quốc ngữ, lỗi gõ |
| Thống kê nội bộ corpus | Xác suất, không cần tri thức ngoài | Mâu thuẫn nhãn, cách đọc lạc loài |
| Tri thức từ điển ngoài | Đối chiếu tri thức | Gán sai chữ cho âm |

Bốn nguồn này bổ sung nhau: nguồn 1–2 chắc chắn nhưng nông, nguồn 4 sâu nhưng phụ thuộc độ phủ từ điển, nguồn 3 không cần từ điển nên vẫn hoạt động cho những chữ hiếm mà không từ điển nào có.

**Thư viện sử dụng:** `pandas` (đọc/nhóm dữ liệu), `openpyxl` + `xlsxwriter` (đọc và ghi Excel), `unicodedata` (chuẩn hoá NFC, phân loại ký tự), `re`, `collections`, `urllib.request`, `json`, `zipfile` — đều là thư viện chuẩn hoặc có sẵn trên Colab.

---

## 2. Tầng 1 — Kiểm tra hình thức Unicode

### 2.1 Xác minh codepoint ↔ ký tự

Đây là phép kiểm tra chặt chẽ nhất trong toàn bộ hệ thống, vì nó có đáp án đúng/sai tuyệt đối:

```python
u = str(uni).strip().upper().replace('U+', '').replace('0X', '')
if not re.fullmatch(r'[0-9A-F]{4,6}', u):      # sai định dạng
if chr(int(u, 16)) != lab:                      # lệch ký tự
```

Ba phép kiểm tra được tách riêng để thông báo lỗi chính xác:

1. **Định dạng mã**: chấp nhận 4–6 chữ số hex, chuẩn hoá cả dạng `U+9081`, `9081`, `0x9081`. Yêu cầu 6 chữ số là bắt buộc vì dữ liệu chứa chữ ngoài mặt phẳng cơ bản (`U+215F6` = 𡗶, `U+2029A` = 𠊚) — biểu thức chỉ chấp nhận 4 chữ số sẽ báo lỗi giả hàng loạt trên chữ Nôm thuần.

2. **Độ dài `label`**: `len(lab) != 1` bị coi là lỗi. Điểm cần lưu ý về kỹ thuật: Python 3 lưu chuỗi theo codepoint (UCS-4) nên `len('𠊚') == 1` dù ký tự này nằm ở mặt phẳng bổ sung. Nếu chuyển logic này sang JavaScript hoặc Java — vốn dùng UTF-16 — thì `'𠊚'.length === 2` do cặp thay thế (surrogate pair) và toàn bộ chữ Ext B sẽ bị báo lỗi sai. Trong dữ liệu của bạn có 2.807 dòng thuộc Ext B, nên đây không phải chi tiết lý thuyết.

3. **Khớp ký tự**: `chr(int(u, 16))` phải bằng đúng `label`.

### 2.2 Phân loại khối Unicode

Bảng tra 14 khoảng codepoint được duyệt tuyến tính cho mỗi ký tự:

```
CJK URO            U+4E00–U+9FFF     Kangxi Radicals    U+2F00–U+2FDF
CJK Ext A          U+3400–U+4DBF     Ext B  U+20000–U+2A6DF
Ext C, D, E, F, G, H, I                Compatibility Ideographs U+F900–U+FAFF
```

Mục đích là hai điều. Thứ nhất, phát hiện ký tự lọt vào không phải chữ Hán — chữ Latin trông giống, dấu câu, ký tự riêng tư (Private Use Area). Thứ hai, cho biết cấu trúc bộ dữ liệu: dữ liệu của bạn có 49.293 dòng URO, 2.807 Ext B, 316 Ext A, phần còn lại rải rác ở Ext C/E/F/G. Tỉ lệ Ext B cao là dấu hiệu đây thực sự là văn bản Nôm chứ không phải Hán văn, vì chữ Nôm tự tạo tập trung ở khối này.

Cần lưu ý về nguy cơ ngầm: chữ ở Compatibility Ideographs (U+F900–U+FAFF) là bản trùng lặp của chữ trong URO, được đưa vào Unicode chỉ để tương thích ngược với bảng mã cũ. Chuẩn hoá NFC/NFKC sẽ tự động đổi chúng về dạng URO, nghĩa là mã lưu trong file và mã sau khi xử lý có thể khác nhau. Đây là nguồn lỗi âm thầm khi ghép dữ liệu từ nhiều nguồn.

### 2.3 Chuẩn hoá và ký tự ẩn

```python
unicodedata.normalize('NFC', v) != v                    # chưa chuẩn NFC
unicodedata.category(ch) in ('Cf', 'Cc')                # ký tự điều khiển/định dạng
ch in '\ufe0f\ufe0e'                                     # variation selector
```

Với cột `syllable`, chữ quốc ngữ có dấu tồn tại ở hai dạng: dựng sẵn (`ề` = một codepoint U+1EC1) và tổ hợp (`ề` = `e` + U+0302 + U+0300, ba codepoint). Hai dạng hiển thị y hệt nhau trên màn hình nhưng **không bằng nhau** khi so chuỗi. Nếu dữ liệu trộn cả hai dạng thì mọi phép `groupby`, `join`, đếm tần suất đều sai lệch mà không có dấu hiệu gì. Kiểm tra NFC vì thế là điều kiện tiên quyết cho tất cả các tầng phía sau.

Nhóm `Cf` bắt zero-width joiner, zero-width space, byte order mark; nhóm `Cc` bắt ký tự điều khiển sót lại từ khâu OCR hoặc copy-paste. Variation selector (U+FE0E/U+FE0F) bị bắt riêng vì nó rất hay đi kèm chữ Hán khi dữ liệu đi qua font hoặc trình soạn thảo có xử lý biến thể tự dạng.

---

## 3. Tầng 2 — Mô hình âm tiết tiếng Việt

### 3.1 Bỏ dấu thanh bằng bảng dịch

```python
TONE_TABLE = str.maketrans('àáảãạăằắẳ...', 'aaaaaăăăă...')
```

Dùng `str.translate` với bảng ánh xạ 60 ký tự thay vì cách phổ biến là `unicodedata.normalize('NFD', s)` rồi lọc dấu phụ. Lý do: cách NFD sẽ bóc luôn cả dấu phụ **tạo chữ cái** trong tiếng Việt, biến `ă` thành `a`, `ơ` thành `o`, `ê` thành `e`. Nhưng `ă` và `a` là hai nguyên âm khác nhau, không phải cùng một nguyên âm khác thanh điệu. Bảng dịch tay giữ nguyên `ă â ê ô ơ ư đ` và chỉ bỏ đúng năm dấu thanh, đây là điều kiện bắt buộc để tầng so sánh vần phía sau không bị nhiễu.

Thanh điệu được tách riêng thành số 0–5 (ngang, huyền, sắc, hỏi, ngã, nặng) qua bảng `TONE_OF`, phục vụ việc phân biệt "trùng âm khác thanh" với "khác âm".

### 3.2 Tách âm tiết

Âm tiết quốc ngữ được tách thành âm đầu và vần theo nguyên tắc **khớp dài nhất** (longest match) trên danh sách 27 âm đầu, sắp xếp sao cho tổ hợp ba chữ đứng trước tổ hợp hai chữ, tổ hợp hai chữ đứng trước phụ âm đơn:

```
ngh  ng nh ch tr th ph kh gh gi qu   b c d đ g h k l m n p r s t v x
```

Nếu không sắp thứ tự này, `nghe` sẽ bị tách thành `ng` + `he` thay vì `ngh` + `e`.

Có một trường hợp ngoại lệ mà nguyên tắc khớp dài nhất xử lý sai, và đây là lỗi đã lộ ra khi chạy thử trên chính dữ liệu của bạn: chữ **gìn**. Máy tách thành `gi` + `n`, phần vần chỉ còn phụ âm `n`, không có nguyên âm, nên báo "âm tiết không hợp lệ" — trong khi `gìn` hoàn toàn đúng chính tả (`g` + `ìn`). Nguyên nhân là chữ `i` trong `gi` khi đứng trước phụ âm cuối lại đóng vai trò nguyên âm chứ không phải một nửa của âm đầu. Luật bổ sung:

```python
if o in ('gi', 'qu') and not (set(rest) & VOWELS):
    return o[0], o[1] + rest        # gi + n  ->  g + in
```

Cùng logic áp cho `qu`. Sau khi sửa, số âm tiết bị báo bất thường trong dữ liệu giảm từ 4 xuống 0 — tức là toàn bộ đều là dương tính giả.

### 3.3 Kiểm tra tính hợp lệ của âm tiết

Bốn ràng buộc, áp dụng sau khi tách:

- Toàn bộ ký tự phải nằm trong bảng chữ cái quốc ngữ (bắt ký tự Latin lạ, chữ số, dấu câu lọt vào)
- Phần vần không rỗng
- Hạt nhân (phần còn lại sau khi bóc phụ âm cuối) chỉ gồm `a ă â e ê i o ô ơ u ư y`
- Hạt nhân không quá 3 ký tự (tiếng Việt không có nguyên âm bốn ký tự)

Phụ âm cuối được nhận dạng từ danh sách `nh ng ch c m n p t i y o u`, cũng theo nguyên tắc khớp dài nhất.

---

## 4. Tầng 3 — Mô hình mượn âm chữ Nôm

Đây là phần cốt lõi và cũng là phần mang tính chuyên ngành nhất.

### 4.1 Vấn đề

Chữ Nôm mượn chữ Hán để ghi âm tiếng Việt, nhưng âm đọc thường **không trùng** âm Hán Việt của chữ đó. 邁 âm Hán Việt là *mại*, trong văn Nôm đọc là *mươi*. 蔑 là *miệt*, đọc là *một*. 朱 là *chu*, đọc là *cho*. Nếu chỉ đối chiếu đúng/sai với từ điển âm Hán Việt thì 64% số cặp trong dữ liệu của bạn sẽ bị báo sai, trong khi phần lớn chúng đúng.

Vì vậy cần một **thước đo mức độ hợp lý ngữ âm**, chứ không phải phép so bằng.

### 4.2 Công thức

```
score(âm_Nôm, âm_Hán_Việt) = 0.5 × onset_sim + 0.5 × rhyme_sim
```

Trọng số 50/50 phản ánh thực tế trong mượn âm Nôm: chữ mượn thường giữ được một trong hai thành phần và biến đổi thành phần kia. Giữ được cả hai (chỉ khác thanh điệu) là trường hợp lý tưởng, được xử lý riêng.

Thang điểm cuối:

| Điểm | Ý nghĩa | Ví dụ |
|---|---|---|
| 1.00 | Trùng khít | |
| 0.95 | Trùng âm, khác thanh điệu | 連 *liên* → *liền* |
| 0.60 | Trùng một thành phần, thành phần kia thuộc cùng nhóm | 朱 *chu* → *cho* (0.8), 除 *trừ* → *giờ* |
| 0.30 | Chỉ gần giống một phần | |
| 0.00 | Không liên hệ ngữ âm | 垩 *ác* → *thánh* |

### 4.3 Nhóm tương đương âm đầu

Không phải chọn tuỳ tiện, mà dựa trên **tương ứng ngữ âm lịch sử tiếng Việt**:

```python
('c','k','q','qu','kh','g','gh')   # cùng gốc âm gốc lưỡi
('d','gi','r','v','nh')            # nhóm đã hợp nhất trong tiếng Việt hiện đại
('tr','ch','t','th')
('s','x','ch')
('ng','ngh')
('t','th','đ','d')
('l','n','nh','r')
('b','m','ph','v')                 # b/ph: âm môi, ph < *p lịch sử
('h','','kh')                      # âm đầu zero
('tr','l','s','gi')                # tổ hợp phụ âm *tl, *bl thời tiền Nôm
('nh','l','d')
('ph','b','v')
```

Nhóm áp chót đáng nói riêng. Tiếng Việt cổ có tổ hợp phụ âm đầu `*tl`, `*bl`, `*ml` mà chữ Nôm ghi lại bằng cách mượn chữ Hán có âm đầu `l`. Về sau các tổ hợp này rụng thành `tr`, `gi`, `nh`, `s`. Đó là lý do 連 *liên* dùng để ghi *trên*, hay 羅 *la* ghi *ra*. Nếu không có nhóm này, hệ thống báo sai 155 dòng chữ 連 trong dữ liệu của bạn.

Cách cài đặt: các nhóm chồng lấn nhau (`ch` có mặt ở cả nhóm `tr/ch/t/th` lẫn `s/x/ch`), nên chúng được gộp vào một `defaultdict(set)` — quan hệ tương đương này **không có tính bắc cầu**, và điều đó là cố ý. Nếu ép bắc cầu, các nhóm sẽ nối thành một khối duy nhất và mọi âm đầu đều tương đương với mọi âm đầu.

Điểm số phân biệt trùng khít (1.0) với cùng nhóm (0.6), nên "giống hệt" vẫn được xếp trên "có thể thay thế".

### 4.4 Chuẩn hoá vần — thứ tự phép biến đổi là mấu chốt

```python
DIPHTHONGS = [('yê','i'), ('iê','i'), ('ia','i'), ('ya','i'),
              ('uô','u'), ('ua','u'), ('ươ','ư'), ('ưa','ư')]

def _norm_rhyme(r):
    for a, b in DIPHTHONGS:           # bước 1: nguyên âm đôi
        r = r.replace(a, b)
    r = re.sub(r'(ng|nh|m)$', 'n', r)  # bước 2: phụ âm cuối
    r = re.sub(r'(c|p|ch)$', 't', r)
    for a, b in [('ă','a'), ('â','a'), ('ê','e'), ('ô','o'),
                 ('ơ','o'), ('ư','u'), ('e','i'), ('u','o')]:   # bước 3: nguyên âm đơn
        r = r.replace(a, b)
    return r
```

Thứ tự ba bước không thể đảo. Bản đầu tiên của script gộp chung nguyên âm đôi và nguyên âm đơn vào một vòng lặp, kết quả là vần `iên` bị bước `ê→e` xử lý trước, thành `ien`, khiến luật `iê→i` không còn khớp; trong khi `ên` thành `en` rồi `in`. Hai vần đáng lẽ tương đương lại ra hai kết quả khác nhau, và cặp 連 *liên* → *trên* bị chấm 0 điểm. Sau khi tách thứ tự, cặp này được 0.6 điểm và xếp đúng vào nhóm mượn âm hợp lý.

Ba nhóm biến đổi tương ứng ba hiện tượng khác nhau: nguyên âm đôi trong tiếng Việt là một đơn vị âm vị nên phải xử lý nguyên khối; phụ âm cuối gộp theo vị trí cấu âm (`ng/nh/m → n` là âm mũi, `c/p/ch → t` là âm tắc); nguyên âm đơn gộp theo khoảng cách trên biểu đồ nguyên âm.

---

## 5. Tầng 4 — Kiểm tra theo nghĩa tiếng Việt

### 5.1 Vấn đề

Ngoài mượn âm, chữ Nôm còn có lối **đọc theo nghĩa** (huấn độc): mượn chữ Hán rồi đọc bằng từ thuần Việt tương ứng nghĩa, bỏ qua hoàn toàn âm đọc. 黄 *hoàng* đọc là *vàng*, 君 *quân* đọc là *vua*, 察 *sát* đọc là *xét*. Với những cặp này, điểm ngữ âm bằng 0 nhưng nhãn hoàn toàn đúng.

### 5.2 Phương pháp

Kỹ thuật dùng ở đây là **truy hồi túi từ có trọng số nghịch tần suất** (bag-of-words retrieval với trọng số kiểu IDF) trên lời giải nghĩa tiếng Việt của từ điển Hán Nôm:

1. Tải từ điển có phần `detail` là lời giải nghĩa bằng tiếng Việt (12.331 chữ).
2. Tách từ toàn bộ kho giải nghĩa bằng regex trên bảng chữ cái quốc ngữ, dựng `collections.Counter` đếm **số mục từ** chứa mỗi từ (document frequency, không phải tổng số lần xuất hiện).
3. Với mỗi cặp (chữ, âm), kiểm tra âm quốc ngữ có nằm trong lời giải nghĩa của chữ đó không.
4. Chấm điểm theo hai yếu tố:

```python
rate = self.df[syllable] / self.n
if syllable in STOP_WORDS or rate > 0.05:
    return 0.2, '(từ quá thông dụng)'
first = low.split('##')[0]                    # đoạn nghĩa gốc
score = 0.9 if syllable in tokens(first) else 0.7
if rate > 0.03: score -= 0.2
```

**Yếu tố nghịch tần suất.** Từ *vàng* xuất hiện trong 1,05% số mục từ, *thánh* trong 0,22%, *khăn* trong 0,21% — chúng mang nhiều thông tin. Ngược lại *là* có mặt trong 35,4% mục từ, *một* trong 17,4%, *cho* trong 8,0%. Nếu không phạt, mọi cặp có âm là *là*, *một*, *cho* đều được xác nhận sai. Đây chính là ý tưởng IDF trong tìm kiếm thông tin, chỉ khác là dùng ngưỡng cứng 5% thay vì hàm log liên tục, vì đầu ra cần là phán quyết rời rạc chứ không phải điểm xếp hạng.

**Danh sách chặn.** 53 từ ngữ pháp được loại thẳng bất kể tần suất: *là, mà, và, thì, cho, của, được, một, những, các, có, không, trong, ngoài...*. Chúng vừa là hư từ tiếng Việt vừa xuất hiện dày trong văn phong định nghĩa từ điển, nên luôn là bằng chứng giả.

**Trọng số vị trí.** Từ điển Hán Nôm dùng `##` để ngăn các nghĩa của một chữ, nghĩa gốc luôn đứng đầu. Từ xuất hiện trong đoạn đầu được 0.9, ở các nghĩa phụ được 0.7. Chi tiết này quan trọng vì lời giải nghĩa dài thường chứa ví dụ, điển tích, trích thơ — nơi bất kỳ từ tiếng Việt nào cũng có thể xuất hiện ngẫu nhiên.

5. Trích dẫn chứng: cắt câu chứa từ khớp, đưa vào cột `dan_chung_nghia` để người rà soát kiểm chứng ngay mà không phải mở từ điển. Ví dụ với 黄/*vàng*, dẫn chứng là "Sắc vàng, sắc ngũ cốc chín".

### 5.3 Điều đã học được khi hiệu chỉnh

Bản chạy đầu tiên xếp 麻 *ma* → *mà* vào nhóm đọc theo nghĩa, vì chữ *mà* có trong lời giải nghĩa của 麻. Đây là dương tính giả kép: *mà* là hư từ, và cặp này thực ra chỉ là khác thanh điệu. Hai chỉnh sửa: đưa *mà* vào danh sách chặn, và **đổi thứ tự cây quyết định** để phép kiểm tra thanh điệu chạy trước phép kiểm tra nghĩa. Sau đó nhóm "đọc theo nghĩa" giảm từ 123 xuống 95 cặp, và toàn bộ 95 cặp còn lại đều đúng khi kiểm tra tay: 法 *phép*, 代 *đời*, 眉 *mày*, 理 *lẽ*, 赦 *tha*, 賊 *giặc*, 敢 *dám*, 君 *vua*.

---

## 6. Tầng 5 — Thống kê nội bộ

Hai phép kiểm tra không cần bất kỳ từ điển nào, nên vẫn có tác dụng với chữ Nôm hiếm nằm ngoài mọi từ điển.

### 6.1 Phát hiện mâu thuẫn

```python
df.groupby(['ocr_char','syllable'])['label'].nunique() > 1
df.groupby('label')['unicode'].nunique() > 1
```

Cùng một ký tự OCR đọc cùng một âm mà bị gán hai chữ khác nhau là mâu thuẫn nội tại — chắc chắn có ít nhất một dòng sai, bất kể từ điển nói gì. Tương tự, một chữ mà mang hai mã Unicode khác nhau là bất khả thi. Đây là loại lỗi có thể khẳng định 100% chỉ bằng dữ liệu, không cần tri thức ngoài. Dữ liệu của bạn không có trường hợp nào.

### 6.2 Phát hiện cách đọc lạc loài

```python
tỉ_lệ = số_lần_của_cặp / tổng_số_lần_của_chữ
cờ_đỏ nếu tỉ_lệ < 2% và cách_đọc_phổ_biến_nhất ≥ 20 lần
```

Đây là phát hiện điểm dị thường dựa trên phân phối tần suất. Lập luận: một chữ có cách đọc chính lặp lại hàng trăm lần, tự nhiên xuất hiện đúng một lần với cách đọc khác hẳn — xác suất cao là lỗi gõ hoặc lỗi thao tác, không phải hiện tượng ngôn ngữ. Chữ 麻 trong dữ liệu của bạn xuất hiện 1.657 lần, trong đó đúng 1 lần được gán âm *nên*. Điều kiện thứ hai (cách đọc chính phải ≥20 lần) tránh báo động với chữ hiếm chỉ có vài lần xuất hiện, nơi phân phối tần suất chưa đủ để kết luận gì.

Phương pháp này bắt được loại lỗi mà đối chiếu từ điển bỏ sót: cách đọc *nên* của chữ 麻 nếu đứng riêng có thể vẫn "hợp lý ngữ âm" ở mức nào đó, chỉ khi đặt cạnh 1.656 lần đọc *ma* mới lộ ra là bất thường.

---

## 7. Cây quyết định tổng hợp

Chín phán quyết A–I, xét theo thứ tự ưu tiên, dừng ở nhánh đầu tiên khớp:

```
A. Âm có trong từ điển Nôm                        -> đúng
B. Âm trùng âm Hán Việt                           -> đúng
D. Điểm ngữ âm ≥ 0.9 (chỉ khác thanh điệu)        -> đúng
C. Điểm khớp nghĩa ≥ 0.6                          -> đúng, đọc theo nghĩa
E. Điểm ngữ âm ≥ 0.5                              -> đúng, mượn âm
F. Điểm khớp nghĩa 0.2–0.6                        -> nên xem
G. Điểm ngữ âm ≥ 0.3                              -> nên xem lại
H. Có trong từ điển nhưng không giải thích được   -> RÀ TAY
I. Chữ không có trong từ điển tham chiếu          -> máy không kết luận được
```

Thứ tự này phản ánh độ tin cậy của bằng chứng chứ không phải thứ tự bảng chữ cái (D đứng trước C là chủ ý — bằng chứng ngữ âm mạnh, tất định, được ưu tiên hơn bằng chứng ngữ nghĩa vốn là suy đoán thống kê).

Điểm quan trọng của thiết kế: nhóm **H tách bạch với I**. H là "có đầy đủ dữ liệu tham chiếu mà vẫn không giải thích nổi" — nghi ngờ mạnh, ưu tiên rà cao nhất. I là "không có gì để đối chiếu" — máy không biết, không phải máy kết luận là sai. Gộp chung hai nhóm sẽ chôn 150 trường hợp đáng ngờ thật vào giữa 205 trường hợp vô can.

---

## 8. Dữ liệu tham chiếu

| Nguồn | Nội dung | Quy mô | Độ phủ trên dữ liệu của bạn |
|---|---|---|---|
| `nomhan-for-yomitan/han_raw.txt` | Âm Hán Việt, đánh chỉ số theo codepoint từ U+4E00 | 11.153 chữ | 82,0% số chữ |
| `rime-chunom/chu_nom.dict.yaml` | Âm Nôm (bộ gõ Rime) | 920 chữ | 26,9% số chữ |
| `tu-dien-han-nom/kanji.json` | Lời giải nghĩa tiếng Việt kiểu Thiều Chửu + âm Hán Việt | 12.331 chữ | 80,7% số chữ |
| **Gộp cả ba** | | | **86,4% số chữ, 92,9% số dòng** |

Nguồn thứ nhất có định dạng đặc biệt: mỗi dòng ứng với một codepoint liên tiếp bắt đầu từ U+4E00, dòng trống nghĩa là chữ đó không có âm. Vì vậy phải đọc tuần tự và đếm dòng, không được bỏ qua dòng trống — bỏ một dòng trống là toàn bộ phần sau lệch một codepoint.

Nguồn thứ ba nặng khoảng 25MB vì mỗi mục từ kèm hình ảnh SVG thứ tự nét; script chỉ đọc trường `detail` và `mean`, phần SVG bị bỏ qua nhưng vẫn phải tải về.

Ngoài ra script có sẵn hàm đọc trường `kVietnamese` trong cơ sở dữ liệu Unihan của Unicode Consortium, hiện để ở dạng chú thích. Bật lên sẽ tăng độ phủ với chữ Ext B, đổi lại thêm khoảng 8MB tải về.

Toàn bộ được tải qua `urllib.request` với `User-Agent` giả lập trình duyệt, bọc trong `try/except` — mất mạng thì các tầng 1, 2, 5 vẫn chạy bình thường, chỉ tầng đối chiếu từ điển bị bỏ qua.

---

## 9. Hiệu năng

| Công đoạn | Thời gian |
|---|---|
| Đọc file Excel 59.371 dòng | ~3 giây |
| Kiểm tra hình thức (tầng 1, 2) | ~4 giây |
| Tải 3 từ điển | ~30–60 giây tuỳ mạng |
| Dựng chỉ mục tần suất từ trên 12.331 mục giải nghĩa | ~1 giây |
| Đối chiếu 2.245 cặp | dưới 1 giây |
| Ghi báo cáo 9 sheet | ~5 giây |

Độ phức tạp: các phép kiểm tra hình thức là O(n) theo số dòng; phần đối chiếu là O(P × R) với P = 2.245 cặp khác nhau và R ≤ 12 âm tham chiếu mỗi chữ. Điểm mấu chốt về hiệu năng là **gộp theo cặp chữ–âm trước khi đối chiếu**: 59.371 dòng chỉ chứa 2.245 cặp khác nhau, nên việc đối chiếu từ điển giảm 26 lần khối lượng. Kết quả sau đó được ánh xạ ngược về từng dòng khi cần thống kê.

Toàn bộ chạy trên CPU, RAM đỉnh khoảng 500MB (chủ yếu do file JSON 25MB đã giải mã).

---

## 10. Giới hạn cần biết

**Đây là hệ thống gợi ý, không phải trọng tài.** Mọi phán quyết A–E chỉ có nghĩa "giải thích được bằng tri thức hiện có", không chứng minh nhãn đúng. Ngược lại H và I không chứng minh nhãn sai.

Các nguồn sai sót đã biết:

- **Bỏ sót (âm tính giả).** Một chữ gán sai nhưng chữ sai lại tình cờ có âm gần giống chữ đúng sẽ lọt qua tầng ngữ âm. Đây là kịch bản có thật với lỗi OCR, vì chữ hình gần nhau đôi khi cũng là chữ đồng thanh phù.
- **Báo nhầm (dương tính giả).** Nhóm I gồm 205 cặp mà máy không có căn cứ; phần lớn là chữ Nôm tự tạo hợp lệ, chỉ là không từ điển nào phủ tới.
- **Không có ngữ cảnh.** Hệ thống xét từng cặp chữ–âm cô lập, không nhìn câu. Một chữ có nhiều cách đọc hợp lệ mà bị gán sai cách đọc trong ngữ cảnh cụ thể thì không thể phát hiện được bằng cách này.
- **Không phân tích cấu tạo chữ.** Chữ Nôm hình thanh gồm thanh phù (gợi âm) và nghĩa phù (gợi nghĩa); hệ thống hiện chỉ xét chữ như một đơn vị nguyên khối.
- **Ngưỡng là kinh nghiệm.** Các con số 0.5, 0.6, 2%, 5%, 20 lần đều hiệu chỉnh bằng tay trên chính bộ dữ liệu này, không phải học từ dữ liệu có nhãn vàng. Đổi sang corpus khác thể loại nên hiệu chỉnh lại.

## 11. Hướng nâng cấp

Theo thứ tự lợi ích trên công sức bỏ ra:

1. **Nạp từ điển Hán Nôm chuyên sâu** nếu có (dạng CSV chữ ↔ âm). Chỉ việc gộp vào biến `nom`, độ phủ tăng thẳng, nhóm I co lại.
2. **Phân tích cấu tạo chữ bằng chuỗi IDS** (Ideographic Description Sequence, có sẵn công khai). Tách được thanh phù thì kiểm tra được: chữ Nôm ghi âm *thánh* mà thanh phù đọc *ác* là bất hợp lý về cấu tạo — một tầng bằng chứng hoàn toàn độc lập với ba tầng hiện có.
3. **Mô hình ngôn ngữ n-gram trên chuỗi âm tiết** của chính corpus, để bắt lỗi phụ thuộc ngữ cảnh mà cách xét cặp cô lập không thấy.
4. **Đối chiếu chéo giữa các bản** nếu cùng một văn bản được số hoá nhiều lần.
