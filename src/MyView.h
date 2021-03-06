#pragma once

/************************************************************************
	 File:        MyView.H (From TrainView.H)

	 Author:
				  Michael Gleicher, gleicher@cs.wisc.edu

	 Modifier
				  Yu-Chi Lai, yu-chi@cs.wisc.edu
				  Maochinn, m10815023@gapps.ntust.edu

	 Comment:
						The TrainView is the window that actually shows the
						train. Its a
						GL display canvas (Fl_Gl_Window).  It is held within
						a TrainWindow
						that is the outer window with all the widgets.
						The TrainView needs
						to be aware of the window - since it might need to
						check the widgets to see how to draw

	  Note:        we need to have pointers to this, but maybe not know
						about it (beware circular references)

	 Platform:    Visio Studio 2019

*************************************************************************/

// Preclarify for preventing the compiler error
class MyWindow;

// The TrainView also needs to handle its user interface events.
// since this is an FLTk Window, we need to include FlTk Headers
// the pragmas turn off the warnings from FlTk - our code should have no warnings,
// but their code doesn't meet that standard
#pragma warning(push)
#pragma warning(disable:4312)
#pragma warning(disable:4311)
#include <Fl/Fl_Gl_Window.h>
#pragma warning(pop)

// this uses the old ArcBall Code
#include "Utilities/ArcBallCam.h"

#include "RenderUtilities/BufferObject.h"
#include "RenderUtilities/Shader.h"
#include "RenderUtilities/Texture.h"
#include "RenderUtilities/CubeMap.h"
#include "RenderUtilities/MeshObject.h"

class GLMesh;

class MyView : public Fl_Gl_Window
{
public:
	// note that we keep the "standard widget" constructor arguments
	MyView(int x, int y, int w, int h, const char* l = 0);

	// overrides of important window things
	virtual int handle(int);
	virtual void draw();

	// setup the projection - assuming that the projection stack has been
	// cleared for you
	void setProjection();

	// Reset the Arc ball control
	void resetArcball();

	//set ubo
	void setUBO();
public:
	ArcBallCam		arcball;			// keep an ArcBall for the UI
	int				selectedCube;  // simple - just remember which cube is selected

	MyWindow* mw;				// The parent of this display window

	Shader* shader = nullptr;
	Texture2D* texture = nullptr;
	VAO* plane = nullptr;
	UBO* commom_matrices = nullptr;

	GLMesh* gl_mesh = nullptr;
	GLMesh* gl_mesh2 = nullptr;
	CubeMap* skybox = nullptr;
	UBO* common_view = nullptr;

	/*for suface splatting*/
	Shader* attribute = nullptr;
	Shader* finalization = nullptr;

	UBO* ubo_Camera = nullptr;
	UBO* ubo_Raycast = nullptr;
	UBO* ubo_Frustum = nullptr;
	UBO* ubo_Parameter = nullptr;

	Sphere* sphere = nullptr;
};