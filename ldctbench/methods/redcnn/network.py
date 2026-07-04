import torch.nn as nn


class Model(nn.Module):
    """Mô hình RED-CNN dùng để khử nhiễu ảnh CT liều thấp.

    - Dạng encoder–decoder đối xứng với 5 lớp tích chập (Conv2d) ở encoder
      và 5 lớp tích chập chuyển vị (ConvTranspose2d) ở decoder.
    - Có các nhánh nối tắt (skip connection) và học phần sai lệch (residual learning):
      mô hình không học ảnh mới từ đầu, mà học phần khác biệt so với ảnh gốc.
    - Gần giống mô hình RED-CNN gốc trong bài báo, chỉ bỏ ReLU cuối để phù hợp với dữ liệu đã chuẩn hóa.
    """

    def __init__(self, args, out_ch=96):
        super(Model, self).__init__()
        # ----- ENCODER: 5 lớp Conv2d liên tiếp (nén thông tin, trích xuất đặc trưng) -----
        # Conv1: từ ảnh CT 1 kênh → out_ch "bản đồ đặc trưng" (feature map), kernel 5x5, stride=1, padding=0
        #        khi không dùng padding, mỗi lần conv làm giảm kích thước ảnh 4 pixel mỗi chiều
        # “out_ch là số ‘con mắt’ khác nhau cùng nhìn ảnh, kernel_size là độ to của mỗi con mắt (nhìn rộng hay hẹp), stride là mỗi lần con mắt nhích bao nhiêu pixel, và padding là có lót thêm viền bên ngoài ảnh hay không để tránh ảnh bị nhỏ lại.”
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        # Conv2–Conv5: giữ nguyên số kênh = out_ch, tiếp tục trích xuất đặc trưng "sâu" hơn từ ảnh
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        # Conv5 đóng vai trò "nút cổ chai" (bottleneck): biểu diễn nén nhất của thông tin ảnh
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        # ----- DECODER: 5 lớp ConvTranspose2d đối xứng (mở rộng lại kích thước, tái tạo ảnh) -----
        self.tconv1 = nn.ConvTranspose2d(
            out_ch, out_ch, kernel_size=5, stride=1, padding=0
        )
        self.tconv2 = nn.ConvTranspose2d(
            out_ch, out_ch, kernel_size=5, stride=1, padding=0
        )
        self.tconv3 = nn.ConvTranspose2d(
            out_ch, out_ch, kernel_size=5, stride=1, padding=0
        )
        self.tconv4 = nn.ConvTranspose2d(
            out_ch, out_ch, kernel_size=5, stride=1, padding=0
        )
        # TConv5: đưa số kênh về lại 1 để thu được ảnh CT xám (grayscale) đã khử nhiễu
        self.tconv5 = nn.ConvTranspose2d(
            out_ch, 1, kernel_size=5, stride=1, padding=0
        )

        # Hàm kích hoạt phi tuyến ReLU: đặt các giá trị âm về 0, giúp mô hình học được các hàm phức tạp hơn
        self.relu = nn.ReLU()

    def forward(self, x):
        # ----- ENCODER -----
        # residual_1: lưu lại ảnh input gốc để dùng cho residual learning ở cuối (output + input)
        residual_1 = x

        # Qua Conv1 + ReLU: bắt đầu trích xuất đặc trưng đơn giản (biên, cạnh, hoa văn nhiễu) từ ảnh đầu vào
        out = self.relu(self.conv1(x))

        # Qua Conv2 + ReLU: thu được đặc trưng sâu hơn, vẫn cùng số kênh
        out = self.relu(self.conv2(out))
        # residual_2: nhánh nối tắt mức "nông", sẽ được cộng lại ở decoder (sau TConv3)
        residual_2 = out

        # Conv3 + Conv4: tiếp tục trích xuất đặc trưng, vùng nhìn (receptive field) của từng neuron ngày càng rộng hơn
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        # residual_3: nhánh nối tắt mức "sâu", sẽ được cộng lại sau TConv1
        residual_3 = out

        # Conv5 (bottleneck): biểu diễn nén nhất của ảnh (đặc trưng trừu tượng nhất)
        out = self.relu(self.conv5(out))

        # ----- DECODER -----
        # TConv1: bắt đầu phóng to lại kích thước không gian của các feature map
        out = self.tconv1(out)
        # Skip connection 1: thêm lại thông tin chi tiết từ residual_3 (sau Conv4)
        out += residual_3

        # TConv2 + TConv3: tiếp tục tái tạo ảnh từ đặc trưng đã được kết hợp
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        # Skip connection 2: thêm lại thông tin từ residual_2 (sau Conv2)
        out += residual_2

        # TConv4: gần đạt kích thước ảnh ban đầu (về mặt chiều rộng và chiều cao)
        out = self.tconv4(self.relu(out))
        # TConv5: ra feature map 1 kênh (cùng kích thước với input)
        out = self.tconv5(self.relu(out))

        # Residual learning: mô hình học phần "sửa lỗi" trên ảnh gốc, nên ở cuối sẽ cộng lại residual_1 (ảnh gốc)
        out += residual_1
        return out
