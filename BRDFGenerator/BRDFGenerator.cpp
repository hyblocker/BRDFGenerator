/* 
 * A Program to generate high quality BRDF lookup tables for the split-sum approximation in UE4s Physically based rendering
 * LUTs are stored in 16 bit or 32 bit floating point textures in either KTX or DDS format. 
 * Written by: Hector Medina-Fetterman
 * 
 * Updates by Hyblocker: Added EXR export
 */

#define _CRT_SECURE_NO_WARNINGS

#include <glm\glm.hpp>
#include <glm\gtc\packing.hpp>

#include <gli\gli.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <miniz.h>
#include <tinyexr.h>

const float PI = 3.14159265358979323846264338327950288;

float RadicalInverse_VdC(unsigned int bits)
{
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return float(bits) * 2.3283064365386963e-10;
}

glm::vec2 Hammersley(unsigned int i, unsigned int N)
{
	return glm::vec2(float(i) / float(N), RadicalInverse_VdC(i));
}

glm::vec3 ImportanceSampleGGX(glm::vec2 Xi, float roughness, glm::vec3 N)
{
	float a = roughness * roughness;

	float phi = 2.0 * PI * Xi.x;
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

	// from spherical coordinates to cartesian coordinates
	glm::vec3 H;
	H.x = cos(phi) * sinTheta;
	H.y = sin(phi) * sinTheta;
	H.z = cosTheta;

	// from tangent-space vector to world-space sample vector
	glm::vec3 up = abs(N.z) < 0.999 ? glm::vec3(0.0, 0.0, 1.0) : glm::vec3(1.0, 0.0, 0.0);
	glm::vec3 tangent = normalize(cross(up, N));
	glm::vec3 bitangent = cross(N, tangent);

	glm::vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
	return normalize(sampleVec);
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
	float a = roughness;
	float k = (a * a) / 2.0;

	float nom = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return nom / denom;
}

float GeometrySmith(float roughness, float NoV, float NoL)
{
	float ggx2 = GeometrySchlickGGX(NoV, roughness);
	float ggx1 = GeometrySchlickGGX(NoL, roughness);

	return ggx1 * ggx2;
}

glm::vec2 IntegrateBRDF(float NdotV, float roughness, unsigned int samples)
{
	glm::vec3 V;
	V.x = sqrt(1.0 - NdotV * NdotV);
	V.y = 0.0;
	V.z = NdotV;

	float A = 0.0;
	float B = 0.0;

	glm::vec3 N = glm::vec3(0.0, 0.0, 1.0);

	for (unsigned int i = 0u; i < samples; ++i)
	{
		glm::vec2 Xi = Hammersley(i, samples);
		glm::vec3 H = ImportanceSampleGGX(Xi, roughness, N);
		glm::vec3 L = normalize(2.0f * dot(V, H) * H - V);

		float NoL = glm::max(L.z, 0.0f);
		float NoH = glm::max(H.z, 0.0f);
		float VoH = glm::max(dot(V, H), 0.0f);
		float NoV = glm::max(dot(N, V), 0.0f);

		if (NoL > 0.0)
		{
			float G = GeometrySmith(roughness, NoV, NoL);

			float G_Vis = (G * VoH) / (NoH * NoV);
			float Fc = pow(1.0 - VoH, 5.0);

			A += (1.0 - Fc) * G_Vis;
			B += Fc * G_Vis;
		}
	}

	return glm::vec2(A / float(samples), B / float(samples));
}

int main(int argc, char** argv)
{
	//Here we set up the default parameters
	int samples = 1024;
	int size = 128;
	int bits = 16;
	std::string filename;

	bool typeIsEXR = false;

	//Must have at least 3 arguments to account for filename
	if (argc < 3) 
	{
		// Inform the user of how to use the program
		std::cout << "Usage: " << argv[0] << " -f filename <Options>\n"
			<< "Options:\n"
			<< "\t-s SIZE \tThe size of the lookup table in pixels [size x size]. Default: 256\n"
			<< "\t-n SAMPLES \tThe number of BRDF samples to integrate per pixel. Default: 1024\n"
			<< "\t-b BITS \tThe number of floating point bits used for texture storage. Can either be 16 or 32. Default: 16\n"
			<< std::endl;

		exit(0);
	}
	else 
	{ 
		//variables for error checking
		errno = 0;
		char* p;

		/* We will iterate over argv[] to get the parameters stored inside.
		 * Note that we're starting on 1 because we don't need to know the
		 * path of the program, which is stored in argv[0] */
		for (int i = 1; i < argc; i++) 
		{ 
			if (i + 1 != argc)
			{
				if (!strcmp(argv[i], "-f")) {
					filename = argv[i + 1];
				}
				else if (!strcmp(argv[i], "-n")) {
					samples = strtol(argv[i + 1], &p, 10);
					if (errno != 0 || *p != '\0' || samples > INT_MAX || samples < 0) {
						std::cout << "Invalid samples input, should be an integer value greater than 0.\n";
						exit(0);
					}
				}
				else if (!strcmp(argv[i], "-s")) {
					size = strtol(argv[i + 1], &p, 10);
					if (errno != 0 || *p != '\0' || size > INT_MAX || size < 0) {
						std::cout << "Invalid size input, should be an integer value greater than 0.\n";
						exit(0);
					}
				}
				else if (!strcmp(argv[i], "-b")) {
					bits = strtol(argv[i + 1], &p, 10);
					if (errno != 0 || *p != '\0' || bits > INT_MAX || (bits != 16 && bits != 32)) {
						std::cout << "Invalid bit input, should be an integer value of 16 or 32.\n";
						exit(0);
					}
				}
			}
		}

		if (!(filename.substr(filename.find_last_of(".") + 1) == "dds") && 
			!(typeIsEXR = filename.substr(filename.find_last_of(".") + 1) == "exr") &&
			!(filename.substr(filename.find_last_of(".") + 1) == "ktx")) 
		{
			std::cout << "Filename must have the dds, ktx or exr extension.\n";
			exit(0);
		}

		if(filename.empty())
		{
			std::cout << "Must provide filename, please try again.\n";
			exit(0);
		}
	}

	if (!typeIsEXR) {

		gli::texture2d tex;
		
		if(bits == 16)
			tex = gli::texture2d(gli::FORMAT_RG16_SFLOAT_PACK16, gli::extent2d(size, size), 1);
		if(bits == 32)
			tex = gli::texture2d(gli::FORMAT_RG32_SFLOAT_PACK32, gli::extent2d(size, size), 1);
		
		for (int y = 0; y < size; y++)
		{
			for (int x = 0; x < size; x++)
			{
				float NoV = (y + 0.5f) * (1.0f / size);
				float roughness = (x + 0.5f) * (1.0f / size);
		
				if (bits == 16)
					tex.store<glm::uint32>({ y, size - 1 - x }, 0, gli::packHalf2x16(IntegrateBRDF(NoV, roughness, samples)));
				if (bits == 32)
					tex.store<glm::vec2>({ y, size - 1 - x }, 0, IntegrateBRDF(NoV, roughness,samples));
			}
		}

		gli::save(tex, filename);

	} else {

		EXRHeader header;
		InitEXRHeader(&header);

		EXRImage image;
		InitEXRImage(&image);

		image.num_channels = 3;

		std::vector<float> images[3];
		images[0].resize(size * size);
		images[1].resize(size * size);
		images[2].resize(size * size);

		for (int i = 0; i < size * size; i++) {
			// Get the current coord from i
			int x = size - i / size;
			int y = i % size;

			float NoV = (y + 0.5f) * (1.0f / size);
			float roughness = (x + 0.5f) * (1.0f / size);

			// Calculate BRDF value
			auto value = IntegrateBRDF(NoV, roughness, samples);

			images[0][i] = value.y;		// R
			images[1][i] = value.x;		// G
			images[2][i] = 0;			// B
		}

		float* image_ptr[3];
		image_ptr[0] = &(images[2].at(0)); // B
		image_ptr[1] = &(images[1].at(0)); // G
		image_ptr[2] = &(images[0].at(0)); // R

		image.images = (unsigned char**)image_ptr;
		image.width = size;
		image.height = size;

		header.num_channels = 3;
		header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
		// Must be (A)BGR order, since most of EXR viewers expect this channel order.
		strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
		strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
		strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';

		header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
		header.requested_pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
		for (int i = 0; i < header.num_channels; i++) {
			header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
			header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
		}

		const char* err = NULL; // or nullptr in C++11 or later.
		int ret = SaveEXRImageToFile(&image, &header, filename.c_str(), &err);
		if (ret != TINYEXR_SUCCESS) {
			// fprintf(stderr, "Save EXR err: %s\n", err);
			std::cerr << "Error while saving LUT to EXR:\n" << err;
			FreeEXRErrorMessage(err); // free's buffer for an error message
			return ret;
		}
		// printf("Saved exr file. [ %s ] \n", filename);

		// free(rgb);

		free(header.channels);
		free(header.pixel_types);
		free(header.requested_pixel_types);
	}

	std::cout << bits << " bit, [" << size << " x " << size << "] BRDF LUT generated using " << samples << " samples.\n";
	std::cout << "Saved LUT to " << filename << ".\n";

    return 0;
}

