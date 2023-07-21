import socket
import time

HEADERSIZE = 10
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1234))

s.listen(5)

while True:
	clientsocket, address = s.accept()
	print("Server ready")
	print(f"Connection from {address} has been established!")
	msg = "welcome to the server!"
	msg = f'{len(msg):<{HEADERSIZE}}' + msg
	clientsocket.send(bytes(msg, "utf-8"))
	




