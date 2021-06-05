# Model-API
### Code mẫu: https://github.com/jurgenarias/Portfolio/blob/master/Voice%20Classification/Code/Speaker_Classifier/Voice_Speaker_Classifier_99.8%25.ipynb

### Model thực hiện quá trình phân loại người nói trên tập dữ liệu gồm 6 người nói, mỗi người có 10 file audio sử dụng mạng softmax

### preprocessor.py
####  Dùng để đọc dữ liệu từ thư mục data, list ra toàn bộ đường dẫn audio trong thư mục và ứng với mỗi người nói được xử lý ứng với đường dẫn audio phù hợp và lưu vào file data để sử dụng sau.

### feature.py
#### Dùng để extract feature từ audio của người nói, các feature được extract: 40 mfcc, chroma feature, tonnezt feature, mel-scaled spectrogram feature, spectral contrast feature sau đó lưu lại để sử dụng cho pha train mô hình.

### train.py
#### Dữ liệu được lưu lại ở pha extract feature đầu tiên được tách ra thành 0.7 - train; 0.2 - validation; 0.1 - test và được chuẩn quá thông qua mô hình Standard Scaler của sklearn trước khi được đưa qua một mô hình mạng softmax để phân lớp ghồm 3 lớp fully-connected relu và 1 lớp fully-connected softmax để phân loại sau đó mô hình được lưu xuống để hỗ trợ cho quá trình viết server.

### server.py
#### Sử dụng form_data để nhận file audio từ client sau đó dùng mô hình đã được train sẵn để dự đoán
