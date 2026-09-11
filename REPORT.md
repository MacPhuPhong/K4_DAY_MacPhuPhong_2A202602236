# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

#### Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): 
- JSON cho ta thấy được ảnh traffic rank 1 gồm có:
- {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "taxonomy_name": "ImageNet-1K",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  }
- rank 1 nghĩa là lớp dự đoán cao nhất trong các lớp dự đoán model biết được trong ảnh, đây chưa chắc là nhãn đúng

#### Record này mô tả toàn ảnh như thế nào?
- theo đề bài là phân loại cấp ảnh nên nó lấy 1 class_name dự đoán được gán nhãn tổng quát, ở đây là "cab" (rank 1), không khoanh 1 chiêc xe riêng nào chỉ là để diễn tả cả ảnh 

#### Ai định nghĩa class list mà checkpoint có thể dự đoán?
- con người định nghĩa lass_list và model học cách dự đoán các checkpoint đã có trong danh sách đó ko thể tự tạo thêm 

#### Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
- ID là dạng cho máy tính hiểu, tên lớp là visual cho con ngườ hiểu và taxonomy là bộ dữ liệu (nhãn). Cùng ID có thể khác taxonomy là manng ý nghĩa khác hoàn toàn.

#### Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
- Guideline cần biết ảnh gán 1 nhãn hay nhiều nhãn, chọn vật thể chủ thể theo kích thước hay vai trò, xử lí ảnh mơ hồ ko rõ mờ, không để tự chọn theo cảm tính

#### Vì sao model score không phải ground truth?
- đơn giản đây chỉ là dự đoán và rank 1 là cab có score = 0,51 là modle đang nghiên về "cab" không có nghĩa là chắn chắn với xác suất 51%
- score cao nhưng có thể là dự ddoassn sai và ngược lại score thấp nhưng cũng có thể dự đoán đúng
- không có class phù hợp trong taxnomy nên là phải chọn class gần nhất

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

#### Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
- 
  {
    "class_id": 5,
    "class_name": "bus",
    "score": 0.912558,
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      93.17,
      187.95,
      223.01,
      320.91
    ],
    "bbox_width": 129.84,
    "bbox_height": 132.96
  }

#### Diễn giải vị trí box bằng lời:
- x_min = 93.17
- y_min = 187.95
- x_max = 223.01
- y_max = 320.91 
- box có tọa độ từ điểm (x_min,y_min) đến điểm (x_max,y_max)

#### So sánh số prediction ở hai threshold:
- threshold thấp thì prediction gồm cả cacs score điểm thấp, ngược lại cao thì prediction score sẽ bắt các socre dự đoán vượt ngưỡng này cho ra it object hơn so với threshold thấp 
- hệ quả là: threshold thấp giúp giảm nguy cơ bỏ sót tăng FP, còn cao làm output ít gọn nhưng có thể mất object thật

#### Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
- threshold thấp: model giữ lại nhiều prediction đồng nghĩa đôn bao phủ tăng nhiều object nhỏ được phat hiện, tuy nhiên sẽ tăng số prediction sai nên cần thời gian cho reviewer kiểm tra 
- threshold cao: ít box cho việc kiểm tra nhưng sẽ dễ bỏ sót vật thể
#### Đề xuất một quy tắc box chặt:
- bouding box cần bao bọc hết vật thể, các cạnh box sát tiếp xúc với điểm ngoài cùng của object

#### Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
- vật thể che khuất thì bao phần nhìn thấy nếu có yêu cầu định lượng vật thì thì có guideline riêng
- Hai object chạm hoặc che nhau có phải tách thành hai box riêng 
- 

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

#### Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):

- Record `kitchen-001` có `class_name = person`, `score = 0.899318` và `polygon_point_count = 348`. Một phần đầu của `polygon_xy` là `[[446, 70], [445, 71], [444, 71], [443, 72], [442, 72], [441, 73], [439, 73], [438, 74]]`. Record này thể hiện model dự đoán một instance người trong ảnh `kitchen`; score là độ tin cậy của model, không phải chất lượng ground truth.

#### Polygon bổ sung chi tiết gì so với box?

- Box chỉ xác định một vùng hình chữ nhật chứa object nên có thể bao gồm cả nền. Polygon gồm nhiều điểm chạy theo đường biên, giúp mô tả hình dạng và vùng pixel của object chi tiết hơn. Trong trường hợp `kitchen-001`, polygon bám theo hình người thay vì lấy toàn bộ nền nằm bên trong box.

#### `instance_id` dùng để làm gì và không phải loại ID nào?

- `instance_id` dùng để phân biệt từng object riêng trong output. Ví dụ, hai người cùng thuộc class `person` vẫn cần hai `instance_id` khác nhau. `instance_id` không phải `class_id`, không phải mã định danh người thật và không phải tracking ID để theo dõi object giữa nhiều frame video.

#### Đề xuất một quy tắc biên mask:

- Mask phải bám sát phần object nhìn thấy được, không lấy dư vùng nền và không nối hai object riêng chỉ vì chúng tiếp xúc hoặc chồng lên nhau. Với các lỗ tự nhiên hoặc khoảng trống nhìn thấy rõ giữa các phần của object, annotator phải xử lý thống nhất theo guideline.

#### Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?

- Guideline cần quy định có vẽ phần object bị che khuất hay chỉ vẽ phần nhìn thấy, cách tách hai object đang tiếp xúc và mức độ rõ tối thiểu để tiếp tục annotation. Nếu không xác định chắc chắn đường biên, class hoặc số lượng instance, annotator phải escalation cho reviewer hoặc Lab Lead thay vì tự suy đoán.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Ảnh thô được annotator gán nhãn theo guideline để tạo ground truth. Ground truth được dùng để huấn luyện model. Sau huấn luyện, model tạo prediction. Reviewer kiểm tra prediction hoặc annotation để phát hiện lỗi; những trường hợp sai hoặc chưa nhất quán được đưa về rework. Prediction chỉ là kết quả của model và không tự động trở thành ground truth.

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một class cho toàn ảnh theo taxonomy và guideline của bài toán single-label | Ảnh `traffic` có nhiều phương tiện nhưng model xếp `cab` hạng 1, nên chủ thể chính của ảnh có thể gây mơ hồ | Xem toàn ảnh và chọn nhãn theo quy tắc xác định chủ thể chính; không chép prediction thành ground truth | Kiểm tra class có đúng taxonomy, đúng phạm vi toàn ảnh và được áp dụng nhất quán hay không |
| Phát hiện vật thể | Một class và một box `bbox_xyxy` cho mỗi object | Trong ảnh `kitchen`, các vật nhỏ như bowl/cup nằm gần nhau, bị che khuất và các nhãn hiển thị chồng lên nhau | Tạo một box riêng cho từng object, vẽ box chặt quanh phần guideline yêu cầu và escalation khi không rõ object | Kiểm tra thiếu/thừa object, nhầm class, box cắt vào object hoặc chứa quá nhiều nền |
| Instance segmentation | Một class và một polygon/mask cho mỗi instance | Biên của người, mặt bàn và các vật nhỏ có vùng tiếp xúc hoặc che khuất nên khó xác định chính xác | Tạo mask riêng cho từng instance và bám sát phần nhìn thấy theo guideline | Kiểm tra mask có dính nền, dính object khác, thiếu vùng object hoặc xử lý vùng che khuất không nhất quán |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ sử dụng ba ảnh công khai được notebook cung cấp; không tải ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hoặc dữ liệu nội bộ lên Colab hay GitHub công khai.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng xử lý, không đưa dữ liệu đó vào report/output và báo cho Lab Coach hoặc Lab Lead để được hướng dẫn.

## 6. Danh sách bằng chứng

- [v] `classification_predictions.json`
- [v] `detection_predictions.json`
- [v] `segmentation_predictions.json`
- [v] `IMAGE_ATTRIBUTION.md`
- [v] `visuals/classification_top5.png`
- [v] `visuals/detection_predictions.png`
- [v] `visuals/segmentation_prediction.png`
- [v] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
