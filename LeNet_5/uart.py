import serial

ser = serial.Serial("COM5", 115200)  # thay COM5 bằng cổng đúng
while True:
    data = ser.readline().decode(errors="ignore").strip()
    print("FPGA:", data)
