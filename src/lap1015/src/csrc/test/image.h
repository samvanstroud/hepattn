#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

class PPMImage
{
public:
	int width, height, max_val;
	unsigned char *raw;
public:
	PPMImage(std::string &fname) : width(0), height(0), max_val(0), raw(0)
	{
		std::ifstream in(fname.c_str(), std::ios::in | std::ios::binary);
		if (in.is_open()) {
			std::string line;

			std::getline(in, line);
			if (line != "P6") {
				std::cout << "\"" << fname << "\" in not a ppm P6 file." << std::endl;
				exit(-1);
			}
			do std::getline(in, line); while (line[0] == '#');
			{
				std::stringstream sline(line);
				sline >> width;
				sline >> height;
			}
			std::getline(in, line);
			{
				std::stringstream sline(line);
				sline >> max_val;
			}

			raw = new unsigned char[width * height * 3];

			in.read((char *)raw, width * height * 3);
		}
		else
		{
			std::cout << "Can't open \"" << fname << "\"" << std::endl;
			exit(-1);
		}
		in.close();
	}
	PPMImage() : width(0), height(0), max_val(0), raw(0) {}
	~PPMImage() { if (raw != 0) delete[] raw; }
};
