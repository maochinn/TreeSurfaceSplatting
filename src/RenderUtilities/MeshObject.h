#pragma once

#include <string>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <glad\glad.h>
#include <glm/glm.hpp>

# include "BufferObject.h"
# include "Sphere.h"

typedef OpenMesh::TriMesh_ArrayKernelT<>  TriMesh;

class MyMesh : public TriMesh
{
public:
	MyMesh();
	~MyMesh();

	int FindVertex(MyMesh::Point pointToFind);
	void ClearMesh();

	void simplification();
};

struct Surfel
{
	Surfel() { }

	Surfel(glm::vec3 c_, glm::vec3 u_, glm::vec3 v_,
		glm::vec3 p_, unsigned int rgba_)
		: c(c_), u(u_), v(v_), p(p_), rgba(rgba_) { }

	glm::vec3 c,	// Position of the ellipse center point.
		u, v,			// Ellipse major and minor axis.
		p;				// Clipping plane.

	unsigned int    rgba;   // Color.
};

class GLMesh
{
public:
	GLMesh();
	~GLMesh();

	bool Init(std::string fileName);
	void Render();
	void LoadTexCoordToShader();

	void exportSurfelPLY(std::string);
	void importSPH(std::string);

	MyMesh mesh;
	std::vector<Surfel> surfels;

	std::vector<glm::vec4> sphere_tree;	// (position.xyz, scale)

	VAO vao;
	VAO surfels_vao;
	//GLuint vao;
	//GLuint ebo;
	//GLuint vboVertices, vboNormal, vboTexCoord;


private:
	std::vector<glm::uvec3> tree_colors;
	void loadTreeColors();

	bool LoadModel(std::string fileName);
	void LoadToShader();
	void MeshToSurfel(std::vector<MyMesh::Point>& vertices, std::vector<unsigned int>& indices, std::vector<Surfel>& surfels, std::vector<glm::uvec3>& tree_colors);
};

