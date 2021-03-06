/************************************************************************
	 File:        MyView.cpp (From MyView.cpp)

	 Author:
				  Michael Gleicher, gleicher@cs.wisc.edu

	 Modifier
				  Yu-Chi Lai, yu-chi@cs.wisc.edu
				  Maochinn, m10815023@gapps.ntust.edu

	 Comment:
						The MyView is the window that actually shows the
						train. Its a
						GL display canvas (Fl_Gl_Window).  It is held within
						a TrainWindow
						that is the outer window with all the widgets.
						The MyView needs
						to be aware of the window - since it might need to
						check the widgets to see how to draw

	  Note:        we need to have pointers to this, but maybe not know
						about it (beware circular references)

	 Platform:    Visio Studio 2019

*************************************************************************/

#include <iostream>
#include <Fl/fl.h>

// we will need OpenGL, and OpenGL needs windows.h
#include <windows.h>
//#include "GL/gl.h"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include "GL/glu.h"

#include "MyView.h"
#include "MyWindow.h"
#include "Utilities/3DUtils.h"


clock_t current_ticks, delta_ticks;
float fps = 0;

std::vector<glm::vec3> random_positions;

//void IdleCallback(void* pData)
//{
//	if (pData != NULL)
//	{
//		current_ticks = clock();
//
//		MyView* trainview = reinterpret_cast<MyView*>(pData);
//
//		delta_ticks = clock() - current_ticks; //the time, in ms, that took to render the scene
//		if (delta_ticks > 0)
//			fps = CLOCKS_PER_SEC / (float)delta_ticks;
//		//system("cls");
//		//std::cout << "FPS:" << fps << std::endl;
//	}
//}

//************************************************************************
//
// * Constructor to set up the GL window
//========================================================================
MyView::
MyView(int x, int y, int w, int h, const char* l)
	: Fl_Gl_Window(x, y, w, h, l)
	//========================================================================
{
	mode(FL_RGB | FL_ALPHA | FL_DOUBLE | FL_STENCIL);

	//Fl::add_idle(IdleCallback, this);

	resetArcball();
}

//************************************************************************
//
// * Reset the camera to look at the world
//========================================================================
void MyView::
resetArcball()
//========================================================================
{
	// Set up the camera to look at the world
	// these parameters might seem magical, and they kindof are
	// a little trial and error goes a long way
	arcball.setup(this, 40, 250, .2f, .4f, 0);
}

//************************************************************************
//
// * FlTk Event handler for the window
//########################################################################
// TODO: 
//       if you want to make the train respond to other events 
//       (like key presses), you might want to hack this.
//########################################################################
//========================================================================
int MyView::handle(int event)
{
	// see if the ArcBall will handle the event - if it does, 
	// then we're done
	// note: the arcball only gets the event if we're in world view
	if (mw->world_cam->value())
		if (arcball.handle(event))
			return 1;

	// remember what button was used
	static int last_push;

	switch (event) {
		// Mouse button being pushed event
	case FL_PUSH:
		last_push = Fl::event_button();
		// if the left button be pushed is left mouse button
		if (last_push == FL_LEFT_MOUSE) {
			//doPick();
			damage(1);
			return 1;
		};
		break;

		// Mouse button release event
	case FL_RELEASE: // button release
		damage(1);
		last_push = 0;
		return 1;

		// Mouse button drag event
	case FL_DRAG:
		break;

		// in order to get keyboard events, we need to accept focus
	case FL_FOCUS:
		return 1;

		// every time the mouse enters this window, aggressively take focus
	case FL_ENTER:
		focus(this);
		break;

	case FL_KEYBOARD:
		int k = Fl::event_key();
		int ks = Fl::event_state();
		if (k == 'p') {

			return 1;
		};
		break;
	}

	return Fl_Gl_Window::handle(event);
}

//************************************************************************
//
// * this is the code that actually draws the window
//   it puts a lot of the work into other routines to simplify things
//========================================================================
void MyView::draw()
{

	//*********************************************************************
	//
	// * Set up basic opengl informaiton
	//
	//**********************************************************************
	//initialized glad
	if (gladLoadGL())
	{
		//initiailize VAO, VBO, Shader...

		std::string common_lib = Shader::readCode("../SurfaceSplatting/src/shaders/common_lib.glsl");
		std::string material_lib = Shader::readCode("../SurfaceSplatting/src/shaders/material_lib.glsl");
		std::string lighting_lib = Shader::readCode("../SurfaceSplatting/src/shaders/lighting.glsl");

		if (!this->shader) {
			this->shader = new Shader(
				common_lib + Shader::readCode("../SurfaceSplatting/src/shaders/simple.vert"),
				std::string(), std::string(), std::string(),
				Shader::readCode("../SurfaceSplatting/src/shaders/simple.frag"));

			this->attribute = new Shader(
				Shader::readCode("../SurfaceSplatting/src/shaders/attribute_vs.glsl") + lighting_lib,
				std::string(), std::string(), std::string(),
				Shader::readCode("../SurfaceSplatting/src/shaders/attribute_fs.glsl"));

			this->finalization = new Shader(
				Shader::readCode("../SurfaceSplatting/src/shaders/finalization_vs.glsl"),
				std::string(), std::string(), std::string(),
				Shader::readCode("../SurfaceSplatting/src/shaders/finalization_fs.glsl") + lighting_lib);
		}
		if (!this->commom_matrices) {
			this->commom_matrices = new UBO();
			this->commom_matrices->size = 3 * sizeof(glm::mat4);
			glGenBuffers(1, &this->commom_matrices->ubo);
			glBindBuffer(GL_UNIFORM_BUFFER, this->commom_matrices->ubo);
			glBufferData(GL_UNIFORM_BUFFER, this->commom_matrices->size, NULL, GL_STATIC_DRAW);
			glBindBuffer(GL_UNIFORM_BUFFER, 0);
		}

		if (!this->gl_mesh)
		{
			this->gl_mesh = new GLMesh();
			//this->gl_mesh->Init("../SurfaceSplatting/Models/Cube.obj");
			//this->gl_mesh->Init("D:/maochinn/NTUST/Master paper/spheretree-1.0-win32/Tree.obj");
			//this->gl_mesh->Init("D:/maochinn/NTUST/Master paper/Sphere Tree/Moduler_tree_leafs.obj");

			this->gl_mesh->Init("../SurfaceSplatting/Models/Alignment_Model.obj");
			this->gl_mesh->importSPH("../SurfaceSplatting/Models/Alignment_Model.sph");
		}
		if (!this->gl_mesh2)
		{
			this->gl_mesh2 = new GLMesh();
			this->gl_mesh2->Init("D:/maochinn/NTUST/Master paper/Sphere Tree/Model_1.sph");
		}

		if (!this->plane) {
			GLfloat  vertices[] = {
				-0.5f ,0.0f , -0.5f,
				-0.5f ,0.0f , 0.5f ,
				0.5f ,0.0f ,0.5f ,
				0.5f ,0.0f ,-0.5f };
			GLfloat  normal[] = {
				0.0f, 1.0f, 0.0f,
				0.0f, 1.0f, 0.0f,
				0.0f, 1.0f, 0.0f,
				0.0f, 1.0f, 0.0f };
			GLfloat  texture_coordinate[] = {
				0.0f, 0.0f,
				1.0f, 0.0f,
				1.0f, 1.0f,
				0.0f, 1.0f };
			GLuint element[] = {
				0, 1, 2,
				0, 2, 3, };

			this->plane = new VAO;
			this->plane->element_amount = sizeof(element) / sizeof(GLuint);
			glGenVertexArrays(1, &this->plane->vao);
			glGenBuffers(3, this->plane->vbo);
			glGenBuffers(1, &this->plane->ebo);

			glBindVertexArray(this->plane->vao);

			// Position attribute
			glBindBuffer(GL_ARRAY_BUFFER, this->plane->vbo[0]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
			glEnableVertexAttribArray(0);

			// Normal attribute
			glBindBuffer(GL_ARRAY_BUFFER, this->plane->vbo[1]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(normal), normal, GL_STATIC_DRAW);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
			glEnableVertexAttribArray(1);

			// Texture Coordinate attribute
			glBindBuffer(GL_ARRAY_BUFFER, this->plane->vbo[2]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(texture_coordinate), texture_coordinate, GL_STATIC_DRAW);
			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (GLvoid*)0);
			glEnableVertexAttribArray(2);

			//Element attribute
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->plane->ebo);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(element), element, GL_STATIC_DRAW);

			// Unbind VAO
			glBindVertexArray(0);
		}

		if (!this->texture)
			//this->texture = new Texture2D("../SurfaceSplatting/Images/church.png");
			this->texture = new Texture2D("D:\\blenderld\\CC_Blender_reconstruction\\Trees\\Tree\\Productions\\Production_2\\Data\\Model\\Model_0.jpg");

		if (!this->skybox) {
			this->skybox = new CubeMap(
				Shader(
					common_lib + Shader::readCode("../SurfaceSplatting/src/shaders/cubeMap.vert"),
					std::string(), std::string(), std::string(),
					material_lib + Shader::readCode("../SurfaceSplatting/src/shaders/cubeMap.frag")),
				"../SurfaceSplatting/Images/skybox/right.jpg",
				"../SurfaceSplatting/Images/skybox/left.jpg",
				"../SurfaceSplatting/Images/skybox/top.jpg",
				"../SurfaceSplatting/Images/skybox/bottom.jpg",
				"../SurfaceSplatting/Images/skybox/back.jpg",
				"../SurfaceSplatting/Images/skybox/front.jpg");
		}

		if (!this->ubo_Camera) {
			this->ubo_Camera = new UBO(3 * sizeof(glm::mat4));
			this->ubo_Raycast = new UBO(sizeof(glm::mat4) + sizeof(glm::vec4));
			this->ubo_Frustum = new UBO(6 * sizeof(glm::vec4));
			this->ubo_Parameter = new UBO(sizeof(glm::vec4) + 4 * sizeof(float));
		}

		if (random_positions.empty())
		{
			for (int i = 0; i < 1000; i++)
			{
				float x = (float)rand() / (RAND_MAX + 1.0);
				float y = (float)rand() / (RAND_MAX + 1.0);
				random_positions.push_back(glm::vec3(x * 100.0f, 0.0f, y * 100.0f));
			}
		}

		if (!this->sphere) {
			this->sphere = new Sphere();
		}

	}
	else
		throw std::runtime_error("Could not initialize GLAD!");

	// Set up the view port
	glViewport(0, 0, w(), h());

	// clear the window, be sure to clear the Z-Buffer too
	glClearColor(1.0f, 1.0f, 1.0f, 0);		// background should be blue

	// we need to clear out the stencil buffer since we'll use
	// it for shadows
	glClearStencil(0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	glEnable(GL_DEPTH);

	// Blayne prefers GL_DIFFUSE
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

	// prepare for projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	setProjection();		// put the code to set up matrices here

	//######################################################################
	// TODO: 
	// you might want to set the lighting up differently. if you do, 
	// we need to set up the lights AFTER setting up the projection
	//######################################################################
	// enable the lighting
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	// top view only needs one light
	if (mw->top_cam->value()) {
		glDisable(GL_LIGHT1);
		glDisable(GL_LIGHT2);
	}
	else {
		glEnable(GL_LIGHT1);
		glEnable(GL_LIGHT2);
	}

	//*********************************************************************
	//
	// * set the light parameters
	//
	//**********************************************************************
	GLfloat lightPosition1[] = { 0,1,1,0 }; // {50, 200.0, 50, 1.0};
	GLfloat lightPosition2[] = { 1, 0, 0, 0 };
	GLfloat lightPosition3[] = { 0, -1, 0, 0 };
	GLfloat yellowLight[] = { 0.5f, 0.5f, .1f, 1.0 };
	GLfloat whiteLight[] = { 1.0f, 1.0f, 1.0f, 1.0 };
	GLfloat blueLight[] = { .1f,.1f,.3f,1.0 };
	GLfloat grayLight[] = { .3f, .3f, .3f, 1.0 };

	glLightfv(GL_LIGHT0, GL_POSITION, lightPosition1);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, whiteLight);
	glLightfv(GL_LIGHT0, GL_AMBIENT, grayLight);

	glLightfv(GL_LIGHT1, GL_POSITION, lightPosition2);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, yellowLight);

	glLightfv(GL_LIGHT2, GL_POSITION, lightPosition3);
	glLightfv(GL_LIGHT2, GL_DIFFUSE, blueLight);



	//*********************************************************************
	// now draw the ground plane
	//*********************************************************************
	// set to opengl fixed pipeline(use opengl 1.x draw function)
	glUseProgram(0);

	setupFloor();
	glDisable(GL_LIGHTING);
	//drawFloor(200,10);


	//*********************************************************************
	// now draw the object and we need to do it twice
	// once for real, and then once for shadows
	//*********************************************************************
	glEnable(GL_LIGHTING);

	setUBO();
	glBindBufferRange(GL_UNIFORM_BUFFER, /*binding point*/0, this->commom_matrices->ubo, 0, this->commom_matrices->size);


	//bind shader
	this->shader->Use();

	glm::mat4 model_matrix = glm::mat4();
	glUniformMatrix4fv(glGetUniformLocation(this->shader->Program, "u_model"), 1, GL_FALSE, &model_matrix[0][0]);
	glUniform3fv(glGetUniformLocation(this->shader->Program, "u_color"), 1, &glm::vec3(0.0f, 1.0f, 0.0f)[0]);
	this->texture->bind(0);
	//this->water->bindWaterTexture(0);
	//this->pool->bindCausticTexture(0);
	glUniform1i(glGetUniformLocation(this->shader->Program, "u_texture"), 0);

	////bind VAO
	//glBindVertexArray(this->plane->vao);
	////glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	//glDrawElements(GL_TRIANGLES, this->plane->element_amount, GL_UNSIGNED_INT, 0);
	////glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	////unbind VAO
	//glBindVertexArray(0);

	//{
	//	glBindVertexArray(this->gl_mesh2->vao.vao);
	//	glDrawElements(GL_TRIANGLES, this->gl_mesh2->mesh.n_faces() * 3, GL_UNSIGNED_INT, 0);
	//	//unbind VAO
	//	glBindVertexArray(0);
	//}

	//{
	//	glBindVertexArray(this->gl_mesh->vao.vao);
	//	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	//	//for (glm::vec3 pos : random_positions)
	//	//{
	//	//	model_matrix = glm::translate(pos);
	//	//	glUniformMatrix4fv(glGetUniformLocation(this->shader->Program, "u_model"), 1, GL_FALSE, &model_matrix[0][0]);
	//	//	glDrawElements(GL_TRIANGLES, this->gl_mesh->mesh.n_faces() * 3, GL_UNSIGNED_INT, 0);
	//	//}
	//	glDrawElements(GL_TRIANGLES, this->gl_mesh->mesh.n_faces() * 3, GL_UNSIGNED_INT, 0);
	//	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//	//unbind VAO
	//	glBindVertexArray(0);
	//}
	glEnable(GL_DEPTH_TEST);
	this->shader->Use();
	for (glm::vec4& sphere : this->gl_mesh->sphere_tree)
	{
		model_matrix = glm::mat4();
		model_matrix = glm::translate(glm::vec3(sphere)) * glm::scale(glm::vec3(sphere.w));
		glUniformMatrix4fv(glGetUniformLocation(this->shader->Program, "u_model"), 1, GL_FALSE, &model_matrix[0][0]);
		this->sphere->render();
	}

	{
			glBindBufferBase(GL_UNIFORM_BUFFER, 0, this->ubo_Camera->ubo);
			glBindBufferBase(GL_UNIFORM_BUFFER, 1, this->ubo_Raycast->ubo);
			glBindBufferBase(GL_UNIFORM_BUFFER, 2, this->ubo_Frustum->ubo);
			glBindBufferBase(GL_UNIFORM_BUFFER, 3, this->ubo_Parameter->ubo);

			this->attribute->Use();

			int num_pts = this->gl_mesh->surfels.size();

			glBindVertexArray(this->gl_mesh->surfels_vao.vao);
			glEnable(GL_DEPTH_TEST);
			glEnable(GL_PROGRAM_POINT_SIZE);
			//for (glm::vec3 pos : random_positions)
			//{
			//	model_matrix = glm::translate(pos);
			//	glUniformMatrix4fv(glGetUniformLocation(this->attribute->Program, "u_model"), 1, GL_FALSE, &model_matrix[0][0]);
			//	glDrawArrays(GL_POINTS, 0, num_pts);
			//}
			model_matrix = glm::mat4();
			glUniformMatrix4fv(glGetUniformLocation(this->attribute->Program, "u_model"), 1, GL_FALSE, &model_matrix[0][0]);
			glDrawArrays(GL_POINTS, 0, num_pts);

			glDisable(GL_PROGRAM_POINT_SIZE);
			glDisable(GL_BLEND);
			glDisable(GL_DEPTH_TEST);
			glBindVertexArray(0);

	}

			//this->gl_mesh->Render();
	

	this->skybox->render();

	//unbind shader(switch to fixed pipeline)
	glUseProgram(0);
}

//************************************************************************
//
// * This sets up both the Projection and the ModelView matrices
//   HOWEVER: it doesn't clear the projection first (the caller handles
//   that) - its important for picking
//========================================================================
void MyView::
setProjection()
//========================================================================
{
	// Compute the aspect ratio (we'll need it)
	float aspect = static_cast<float>(w()) / static_cast<float>(h());

	// Check whether we use the world camp
	if (mw->world_cam->value())
		arcball.setProjection(false);
	// Or we use the top cam
	else if (mw->top_cam->value()) {
		float wi, he;
		if (aspect >= 1) {
			wi = 110;
			he = wi / aspect;
		}
		else {
			he = 110;
			wi = he * aspect;
		}

		// Set up the top camera drop mode to be orthogonal and set
		// up proper projection matrix
		glMatrixMode(GL_PROJECTION);
		glOrtho(-wi, wi, -he, he, 200, -200);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glRotatef(-90, 1, 0, 0);
	}
}

void MyView::setUBO()
{
	float wdt = this->pixel_w();
	float hgt = this->pixel_h();

	glm::mat4 view_matrix;
	glGetFloatv(GL_MODELVIEW_MATRIX, &view_matrix[0][0]);
	//HMatrix view_matrix; 
	//this->arcball.getMatrix(view_matrix);

	glm::mat4 projection_matrix;
	glGetFloatv(GL_PROJECTION_MATRIX, &projection_matrix[0][0]);
	//projection_matrix = glm::perspective(glm::radians(this->arcball.getFoV()), (GLfloat)wdt / (GLfloat)hgt, 0.01f, 1000.0f);


	glBindBuffer(GL_UNIFORM_BUFFER, this->commom_matrices->ubo);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(glm::mat4), &projection_matrix[0][0]);
	glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::mat4), &view_matrix[0][0]);
	glBufferSubData(GL_UNIFORM_BUFFER, 2 * sizeof(glm::mat4), sizeof(glm::mat4), &glm::inverse(view_matrix)[0][0]);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	/**/

	glBindBuffer(GL_UNIFORM_BUFFER, this->ubo_Camera->ubo);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(glm::mat4), &view_matrix[0][0]);
	glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::mat4), &glm::inverse(view_matrix)[0][0]);
	glBufferSubData(GL_UNIFORM_BUFFER, 2 * sizeof(glm::mat4), sizeof(glm::mat4), &projection_matrix[0][0]);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);
	glm::vec4 viewportf(viewport[0], viewport[1], viewport[2], viewport[3]);

	glBindBuffer(GL_UNIFORM_BUFFER, this->ubo_Raycast->ubo);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(glm::mat4), &glm::inverse(projection_matrix)[0][0]);
	glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::vec4), &viewportf[0]);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glm::vec4 frustum_plane[6];
	for (unsigned int i(0); i < 6; ++i)
	{
		frustum_plane[i] = projection_matrix[3] + (-1.0f + 2.0f
			* static_cast<float>(i % 2))* projection_matrix[i / 2];
	}
	for (unsigned int i(0); i < 6; ++i)
	{
		frustum_plane[i] = (1.0f / glm::vec3(frustum_plane[i]).length()) * frustum_plane[i];
	}

	glBindBuffer(GL_UNIFORM_BUFFER, this->ubo_Frustum->ubo);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 6 * sizeof(glm::vec4), &frustum_plane[0]);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glm::vec3 color(0.5f);
	float shininess = 8.0f;
	float radius_scale = mw->radius_slider->value();
	float ewa_radius = 1.0f;
	float epsilon = 1.0f * 1e-3f;

	glBindBuffer(GL_UNIFORM_BUFFER, this->ubo_Parameter->ubo);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 3 * sizeof(float), &color[0]);
	glBufferSubData(GL_UNIFORM_BUFFER, 12, sizeof(float), &shininess);
	glBufferSubData(GL_UNIFORM_BUFFER, 16, sizeof(float), &radius_scale);
	glBufferSubData(GL_UNIFORM_BUFFER, 20, sizeof(float), &ewa_radius);
	glBufferSubData(GL_UNIFORM_BUFFER, 24, sizeof(float), &epsilon);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}