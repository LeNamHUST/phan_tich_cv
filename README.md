# phan_tich_cv
* API: 127.0.0.1:8000/extract_cv
* input_cv 
![input_cv](https://github.com/LeNamHUST/phan_tich_cv/assets/136665511/07cb57dd-ab13-46d0-85a1-d2c1c58362df)
* Sử dụng thuật toán Yolo đã được train với data cv để detect các trường: name, job_title, avatar, infor, block
![detect_image](https://github.com/LeNamHUST/phan_tich_cv/assets/136665511/28114854-c9ea-4b9a-9c6c-c6e27c08e87e)
* Sử dụng thuật toán PaddleOCR để trích xuất text trong các trường.
* output:
'''
"FullName": " NGUYỄN NGỌC HUYỀN",
"DateOfBirth": "26/01/2000",
"Gender": Null,
"Email": "rongdibo260@gmail.com",
"PhoneNumber": " 0908912446",
"Address": "pTam Hòa tp Biên Hoà",
"Education": "\n Trường cấp ba : THPT Lê Quý Đôn\n Trường đại học: Đồng Nai\n Chuyên ngành : Quản trị kinh doanh\n Loại : chưa",
"Job": " TRƯỞNG CA",
"Salary": "",
"Workplace": "",
"Objective": "\n Tìm cơ hội cống hiến cho cửa hàng ,được làm vào\n học hỏi nhiều kinh nghiệm trong khi làm việc team\n phát triển trong môi trường chuyên nghiệp mà\n cửa hàng đã tạo dựng .Cầu tiến trong tương lai\n gắn bó lâu dài .",
"Skill": "\n Xử Lý tình huống với khách hàng\n Khả năng giao tiếp hoạt ngôn\n Kỹ năng làm việc nhóm , lắng nghe\n Kỹ năng kết hợp các loại thức uống\n Kỹ năng decol\n Sử dụng thành thạo các công cụ pha chế\n Kiến thức về tâm lý khách hàng",
"Awards": "\n Nhân viên được yêu thích: giải nhì 2 tháng liên tiếp",
"Certifications": "",
"Hobbies": "\n Sáng tạo\n -Học hỏi\n -Tìm tòi cái mới\n -Du lịch\n -Giao tiếp\n -Thưởng thức các loại thức uống",
"References": "\n Bùi Thị Trúc Linh\n Quản lí cũ\n Xuất+84 93 7780631-",
"Experience": "\n Thời gian từ: 1 / 2018-9 / 2019\n 1.Dj cube\n -Pha chế\n -Phục vụ\n Thời gian từ: 11 / 2019-6 / 2020\n 2Bus cafe\n -Pha chế\n -Thu ngân\n Thời gian từ: 10 / 2020-6 / 2022\n 3.Phòng bánh đen\n -Thợ bánh bông lan\n -Thợ decor\n Thời gian từ: 10 / 2022-3 / 2023\n 4.Hanuel tea\n - Pha chế\n -Trưởng ca",
"Activities": "",
"Project": "\n Đào tạo nhân viên part -time\n Đào tạo kĩ năng xử lí tình huống\n Tham gia đóng góp ý kiến cải tạo cửa hàng\n Xây durna tình đồng đội trong công việc",
"AdditionalInformation": " THÔNG TIN THÊM\n Mong sẽ được hợp tác với nhà tuyển dụng sớm\n nhất và tôi sẽ không làm mọi người thất vọng\n Dạ tôi xin cám ơn"
'''
