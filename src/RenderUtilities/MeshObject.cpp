#include <map>
#include <Eigen/Core>
#include <Eigen/Eigenvalues> 
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <glm/gtx/transform.hpp>


#include "MeshObject.h"

using namespace Eigen;

struct OpenMesh::VertexHandle const OpenMesh::PolyConnectivity::InvalidVertexHandle;

void
steiner_circumellipse(glm::vec3 v0, glm::vec3 v1,
	glm::vec3 v2, glm::vec3& p0, glm::vec3& t1, glm::vec3& t2)
{
	Matrix2f Q;
	Vector3f d0, d1, d2;
	{
		using Vec = Map<const Vector3f>;
		Vec v[] = { Vec(&v0[0]), Vec(&v1[0]), Vec(&v2[0]) };

		d0 = v[1] - v[0];
		d0.normalize();

		d1 = v[2] - v[0];
		d1 = d1 - d0 * d0.dot(d1);
		d1.normalize();

		d2 = (1.0f / 3.0f) * (v[0] + v[1] + v[2]);

		Vector2f p[3];
		for (unsigned int j(0); j < 3; ++j)
		{
			p[j] = Vector2f(
				d0.dot(v[j] - d2),
				d1.dot(v[j] - d2)
			);
		}

		Matrix3f A;
		for (unsigned int j(0); j < 3; ++j)
		{
			A.row(j) = Vector3f(
				p[j].x() * p[j].x(),
				2.0f * p[j].x() * p[j].y(),
				p[j].y() * p[j].y()
			);
		}

		FullPivLU<Matrix3f> lu(A);
		Vector3f res = lu.solve(Vector3f::Ones());

		Q(0, 0) = res(0);
		Q(1, 1) = res(2);
		Q(0, 1) = Q(1, 0) = res(1);

		{
			SelfAdjointEigenSolver<Matrix2f> es;
			es.compute(Q);

			Vector2f const& l = es.eigenvalues();
			Vector2f const& e0 = es.eigenvectors().col(0);
			Vector2f const& e1 = es.eigenvectors().col(1);

			Vector3f p0_ = d2;
			Vector3f t1_ = (1.0f / std::sqrt(l.x())) * (d0 * e0.x() + d1 * e0.y());
			Vector3f t2_ = (1.0f / std::sqrt(l.y())) * (d0 * e1.x() + d1 * e1.y());

			p0 = glm::vec3(p0_.x(), p0_.y(), p0_.z());
			t1 = glm::vec3(t1_.x(), t1_.y(), t1_.z());
			t2 = glm::vec3(t2_.x(), t2_.y(), t2_.z());
		}

	}
}
void hsv2rgb(float h, float s, float v, float& r, float& g, float& b)
{
	float h_i = std::floor(h / 60.0f);
	float f = h / 60.0f - h_i;

	float p = v * (1.0f - s);
	float q = v * (1.0f - s * f);
	float t = v * (1.0f - s * (1.0f - f));

	switch (static_cast<int>(h_i))
	{
	case 1:
		r = q; g = v; b = p;
		break;
	case 2:
		r = p; g = v; b = t;
		break;
	case 3:
		r = p; g = q; b = v;
		break;
	case 4:
		r = t; g = p; b = v;
		break;
	case 5:
		r = v; g = p; b = q;
		break;
	default:
		r = v; g = t; b = p;
	}
}

void
face_to_surfel(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, Surfel& surfel, const std::vector<glm::uvec3>& tree_colors, glm::vec3 color)
{
	glm::vec3 p0, t1, t2;
	steiner_circumellipse(
		v0, v1, v2,
		p0, t1, t2
	);

	glm::vec3 n_s = glm::cross(t1, t2);
	glm::vec3 n_t = glm::cross(v1 - v0, v2 - v0);

	if (glm::dot(n_t, n_s) < 0.0f)
	{
		glm::vec3 temp = t1;
		t1 = t2;
		t2 = temp;
	}
	//float random_angle = 30.0f * glm::radians(rand() / (RAND_MAX + 1.0f));
	//t1 = glm::rotate(random_angle, t2) * glm::vec4(t1, 0.0f);

	surfel.c = p0;
	surfel.u = t1;
	surfel.v = t2;
	surfel.p = glm::vec3(0.0f);

	if (tree_colors.empty())
	{
		float h = std::min((std::abs(p0.x) / 0.45f) * 360.0f, 360.0f);
		float r, g, b;
		hsv2rgb(h, 1.0f, 1.0f, r, g, b);
		surfel.rgba = static_cast<unsigned int>(r * 255.0f)
			| (static_cast<unsigned int>(g * 255.0f) << 8)
			| (static_cast<unsigned int>(b * 255.0f) << 16);
	}
	else
	{
		glm::uvec3 color = tree_colors[rand() % tree_colors.size()];
		color *=  2;
		surfel.rgba = color.r | color.g << 8 | color.b << 16;
		//surfel.rgba = static_cast<unsigned int>(color.r * 255.0f)
		//	| (static_cast<unsigned int>(color.g * 255.0f) << 8)
		//	| (static_cast<unsigned int>(color.b * 255.0f) << 16);
	}

}


#pragma region MyMesh

MyMesh::MyMesh()
{
	request_vertex_normals();
	request_vertex_status();
	request_face_status();
	request_edge_status();

	request_vertex_texcoords2D();
}

MyMesh::~MyMesh()
{

}

int MyMesh::FindVertex(MyMesh::Point pointToFind)
{
	int idx = -1;
	for (MyMesh::VertexIter v_it = vertices_begin(); v_it != vertices_end(); ++v_it)
	{
		MyMesh::Point p = point(*v_it);
		if (pointToFind == p)
		{
			idx = v_it->idx();
			break;
		}
	}

	return idx;
}

void MyMesh::ClearMesh()
{
	if (!faces_empty())
	{
		for (MyMesh::FaceIter f_it = faces_begin(); f_it != faces_end(); ++f_it)
		{
			delete_face(*f_it, true);
		}

		garbage_collection();
	}
}

void MyMesh::simplification()
{

}

#pragma endregion

#pragma region GLMesh

GLMesh::GLMesh()
{
}

GLMesh::~GLMesh()
{

}

bool GLMesh::Init(std::string fileName)
{
	if (LoadModel(fileName))
	{
		loadTreeColors();
		LoadToShader();
		LoadTexCoordToShader();


		return true;
	}
	return false;
}

void GLMesh::Render()
{
	//glBindVertexArray(this->vao.vao);
	//glDrawElements(GL_TRIANGLES, mesh.n_faces() * 3, GL_UNSIGNED_INT, 0);
	//glBindVertexArray(0);

	/**/
	unsigned int num_pts = (unsigned int)this->surfels.size();

	glBindBuffer(GL_ARRAY_BUFFER, this->surfels_vao.vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, 0, NULL, GL_STATIC_DRAW);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Surfel) * num_pts,
		&this->surfels.front(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	glEnable(GL_DEPTH_TEST);
	glEnable(GL_PROGRAM_POINT_SIZE);

	//if (!depth_only && m_soft_zbuffer)
	//{
	//	glEnable(GL_BLEND);
	//	glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
	//	glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_ONE, GL_ONE);
	//}

	//glProgram& program = depth_only ? m_visibility : m_attribute;

	//program.use();
	

	//if (depth_only)
	//{
	//	glDepthMask(GL_TRUE);
	//	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	//}
	//else
	//{
	//	if (m_soft_zbuffer)
	//		glDepthMask(GL_FALSE);
	//	else
	//		glDepthMask(GL_TRUE);

	//	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	//}

	//setup_uniforms(program);

	//if (!depth_only && m_soft_zbuffer && m_ewa_filter)
	//{
	//	glActiveTexture(GL_TEXTURE1);
	//	glBindTexture(GL_TEXTURE_1D, m_filter_kernel);

	//	program.set_uniform_1i("filter_kernel", 1);
	//}

	glBindVertexArray(this->surfels_vao.vao);
	glDrawArrays(GL_POINTS, 0, num_pts);
	glBindVertexArray(0);

	//program.unuse();

	glDisable(GL_PROGRAM_POINT_SIZE);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
}

bool GLMesh::LoadModel(std::string fileName)
{
	OpenMesh::IO::Options ropt = OpenMesh::IO::Options::VertexTexCoord;
	if (OpenMesh::IO::read_mesh(mesh, fileName, ropt))
	{
		if (!ropt.check(OpenMesh::IO::Options::VertexNormal) && mesh.has_vertex_normals())
		{
			mesh.request_face_normals();
			mesh.update_normals();
			mesh.release_face_normals();
		}
		return true;
	}

	return false;
}

void GLMesh::LoadToShader()
{
	std::vector<MyMesh::Point> vertices;
	std::vector<MyMesh::Normal> normals;

	vertices.reserve(mesh.n_vertices());
	normals.reserve(mesh.n_vertices());

	for (MyMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
	{
		vertices.push_back(mesh.point(*v_it));
		normals.push_back(mesh.normal(*v_it));

		//MyMesh::Point p = mesh.point(*v_it);
	}

	std::vector<unsigned int> indices;
	indices.reserve(mesh.n_faces() * 3);
	for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it)
	{
		for (MyMesh::FaceVertexIter fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it)
		{
			indices.push_back(fv_it->idx());
		}
	}
	std::cout << mesh.n_faces() << std::endl;
	glGenVertexArrays(1, &this->vao.vao);
	glBindVertexArray(this->vao.vao);

	glGenBuffers(3, this->vao.vbo);

	glBindBuffer(GL_ARRAY_BUFFER, this->vao.vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(MyMesh::Point) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, this->vao.vbo[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(MyMesh::Normal) * normals.size(), &normals[0], GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);

	glGenBuffers(1, &this->vao.ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->vao.ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * indices.size(), &indices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	this->MeshToSurfel(vertices, indices, this->surfels, this->tree_colors);

	glGenVertexArrays(1, &this->surfels_vao.vao);
	glBindVertexArray(this->surfels_vao.vao);

	glGenBuffers(1, this->surfels_vao.vbo);

	glBindBuffer(GL_ARRAY_BUFFER, this->surfels_vao.vbo[0]);

	// Center c.
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
		sizeof(Surfel), reinterpret_cast<const GLfloat*>(0));

	// Tagent vector u.
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
		sizeof(Surfel), reinterpret_cast<const GLfloat*>(12));

	// Tangent vector v.
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE,
		sizeof(Surfel), reinterpret_cast<const GLfloat*>(24));

	// Clipping plane p.
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE,
		sizeof(Surfel), reinterpret_cast<const GLfloat*>(36));

	// Color rgba.
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 4, GL_UNSIGNED_BYTE, GL_TRUE,
		sizeof(Surfel), reinterpret_cast<const GLbyte*>(48));

	glBindVertexArray(0);

	unsigned int num_pts = (unsigned int)this->surfels.size();
	glBindBuffer(GL_ARRAY_BUFFER, this->surfels_vao.vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, 0, NULL, GL_STATIC_DRAW);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Surfel) * num_pts,
		&this->surfels.front(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLMesh::LoadTexCoordToShader()
{
	if (mesh.has_vertex_texcoords2D())
	{
		std::vector<MyMesh::TexCoord2D> texCoords;
		for (MyMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
		{
			MyMesh::TexCoord2D texCoord = mesh.texcoord2D(*v_it);
			texCoords.push_back(texCoord);
		}

		glBindVertexArray(this->vao.vao);

		glBindBuffer(GL_ARRAY_BUFFER, this->vao.vbo[2]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(MyMesh::TexCoord2D) * texCoords.size(), &texCoords[0], GL_STATIC_DRAW);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(2);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}
}

void
GLMesh::MeshToSurfel(
	std::vector<MyMesh::Point>& vertices, std::vector<unsigned int>& indices, std::vector<Surfel>& surfels, std::vector<glm::uvec3>& tree_colors)
{
	surfels.resize(indices.size()/3);
	std::vector<std::thread> threads(std::thread::hardware_concurrency());
	;
	for (std::size_t i(0); i < threads.size(); ++i)
	{
		std::size_t b = i * surfels.size() / threads.size();
		std::size_t e = (i + 1) * surfels.size() / threads.size();

		threads[i] = std::thread([b, e, &vertices, &indices, &surfels, &tree_colors]() {
			for (std::size_t j = b; j < e; ++j)
			{
				float* v0 = vertices[indices[3*j]].data();
				float* v1 = vertices[indices[3*j + 1]].data();
				float* v2 = vertices[indices[3*j + 2]].data();
				
				face_to_surfel(
					glm::vec3(v0[0], v0[1], v0[2]),
					glm::vec3(v1[0], v1[1], v1[2]),
					glm::vec3(v2[0], v2[1], v2[2]),
					surfels[j], tree_colors, glm::vec3((float)b / surfels.size()));
			}
			});
	}

	for (auto& t : threads) { t.join(); }
}



void 
GLMesh::loadTreeColors()
{
	cv::Mat img;
	//cv::imread("../SurfaceSplatting/Images/Tree.png", cv::IMREAD_COLOR).convertTo(img, CV_32FC4, 1 / 255.0f);	//unsigned char to float
	img = cv::imread("../SurfaceSplatting/Images/Tree.png", cv::IMREAD_UNCHANGED);

	if (img.type() == CV_8UC4)
	{
		for (cv::MatIterator_<cv::Vec4b> it = img.begin<cv::Vec4b>(), end = img.end<cv::Vec4b>(); it != end; ++it)
		{
			if ((*it)[3] > 0)
				this->tree_colors.push_back(glm::uvec3((*it)[2], (*it)[1], (*it)[0]));
		}
	}

	img.release();


}

void
GLMesh::exportSurfelPLY(std::string path)
{
	FILE* fp;

	fp = fopen(path.c_str(), "w+");
	fprintf(fp, "ply\n");
	fprintf(fp, "format ascii 1.0\ncomment NTUST CG LAB generated\n");
	fprintf(fp, "element vertex %d\n", surfels.size());
	fprintf(fp, "property float x\nproperty float y\nproperty float z\n");
	fprintf(fp, "property float nx\nproperty float ny\nproperty float nz\n");
	fprintf(fp, "element face 0\nproperty list uchar int vertex_indices\nend_header\n");

	for (Surfel s : surfels)
	{
		glm::vec3 position = (0.232252f * s.c - glm::vec3(0.122168f, -0.390619f, 0.010802f));
		glm::vec3 normal = glm::normalize(glm::cross(s.u, s.v));
		fprintf(fp, "%f %f %f %f %f %f \n", position.x, position.y, position.z, normal.x, normal.y, normal.z);
	}

	fclose(fp);

	return;
}

void 
GLMesh::importSPH(std::string path)
{
	int temp, amount;
	FILE* fp;

	fp = fopen(path.c_str(), "r");
	fscanf(fp, "%d %d", &temp, &amount);

	for (int i = 0; i < amount; i++)
	{
		glm::vec4 sphere;
		fscanf(fp, "%f %f %f %f", &sphere.x, &sphere.y, &sphere.z, &sphere.w);
		sphere_tree.push_back(sphere);
	}

	fclose(fp);
}

#pragma endregion
