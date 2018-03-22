import sys, os, socket

def main():
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_address = ('localhost', 4321)
	sock.connect(server_address)

	try:
		print("Input an image path: "),
		image_path = raw_input()
		if not image_path:
			print('Invalid path...')
			sys.exit(0)
		sock.send(open(image_path).read())
		print(sock.recv(1024))
	except:
		print('Error')

	sock.close()

if __name__ == '__main__':
	main()