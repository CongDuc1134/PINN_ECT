# Python Execution & Conda Environment Rule

## Rule: Môi Trường Thực Thi Conda Bắt Buộc

Mọi tác vụ thực thi mã nguồn Python, chạy script huấn luyện, suy luận mô hình, kiểm thử hoặc cài đặt thư viện trong dự án này **PHẢI** luôn sử dụng môi trường Conda **`konabi`**.

### Hướng Dẫn Thực Thi
1. **Khi chạy lệnh terminal đơn lẻ**:
   - Sử dụng cú pháp: `conda run -n konabi python <script_path>`
   - Hoặc: `conda activate konabi; python <script_path>`
2. **Khi cài đặt / quản lý gói**:
   - Sử dụng: `conda run -n konabi pip install <package_name>` hoặc `conda install -n konabi <package_name>`
3. **Không chạy trực tiếp bằng python mặc định của hệ thống (`python script.py`) nếu chưa active môi trường `konabi`**.
