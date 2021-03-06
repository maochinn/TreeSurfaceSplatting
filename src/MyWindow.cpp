/************************************************************************
	 File:        MyWindow.cpp (from MyWindow.cpp)

	 Author:
				  Michael Gleicher, gleicher@cs.wisc.edu

	 Modifier
				  Yu-Chi Lai, yu-chi@cs.wisc.edu
				  Maochinn, m10815023@gapps.edu.tw

	 Comment:
						this class defines the window in which the project
						runs - its the outer windows that contain all of
						the widgets, including the "TrainView" which has the
						actual OpenGL window in which the train is drawn

						You might want to modify this class to add new widgets
						for controlling	your train

						This takes care of lots of things - including installing
						itself into the FlTk "idle" loop so that we get periodic
						updates (if we're running the train).


	 Platform:    Visio Studio 2019

*************************************************************************/

#include <FL/fl.h>
#include <FL/Fl_Box.h>

// for using the real time clock
#include <time.h>

#include "MyWindow.h"
#include "MyView.h"
#include "CallBack.h"



//************************************************************************
//
// * Constructor
//========================================================================
MyWindow::
MyWindow(const int x, const int y)
	: Fl_Double_Window(x, y, 800, 600, "Mesh Simplification")
	//========================================================================
{
	// make all of the widgets
	begin();	// add to this widget
	{
		int pty = 5;			// where the last widgets were drawn

		myView = new MyView(5, 5, 590, 590);
		myView->mw = this;
		//trainView->m_pTrack = &m_Track;
		this->resizable(myView);

		// to make resizing work better, put all the widgets in a group
		widgets = new Fl_Group(600, 5, 190, 590);
		widgets->begin();

		//runButton = new Fl_Button(605, pty, 60, 20, "Run");
		//togglify(runButton);

		//Fl_Button* fb = new Fl_Button(700,pty,25,20,"@>>");
		//fb->callback((Fl_Callback*)forwCB,this);
		//Fl_Button* rb = new Fl_Button(670,pty,25,20,"@<<");
		//rb->callback((Fl_Callback*)backCB,this);
		//
		//arcLength = new Fl_Button(730,pty,65,20,"ArcLength");
		//togglify(arcLength,1);
  //
		//pty+=25;
		//speed = new Fl_Value_Slider(655,pty,140,20,"speed");
		//speed->range(0,10);
		//speed->value(2);
		//speed->align(FL_ALIGN_LEFT);
		//speed->type(FL_HORIZONTAL);

		//pty += 30;

		// camera buttons - in a radio button group
		Fl_Group* camGroup = new Fl_Group(600, pty, 195, 20);
		camGroup->begin();
		world_cam = new Fl_Button(605, pty, 60, 20, "World");
		world_cam->type(FL_RADIO_BUTTON);		// radio button
		world_cam->value(1);			// turned on
		world_cam->selection_color((Fl_Color)3); // yellow when pressed
		world_cam->callback((Fl_Callback*)damageCB, this);
		//trainCam = new Fl_Button(670, pty, 60, 20, "Train");
  //      trainCam->type(FL_RADIO_BUTTON);
  //      trainCam->value(0);
  //      trainCam->selection_color((Fl_Color)3);
		//trainCam->callback((Fl_Callback*)damageCB,this);
		top_cam = new Fl_Button(735, pty, 60, 20, "Top");
		top_cam->type(FL_RADIO_BUTTON);
		top_cam->value(0);
		top_cam->selection_color((Fl_Color)3);
		top_cam->callback((Fl_Callback*)damageCB, this);
		camGroup->end();

		pty += 30;

		// browser to select spline types
		simplification_browser = new Fl_Browser(605, pty, 120, 75, "Method");
		simplification_browser->type(2);		// select
		simplification_browser->callback((Fl_Callback*)damageCB, this);
		simplification_browser->add("Average");
		simplification_browser->add("Median");
		simplification_browser->add("Error quadrics");
		simplification_browser->select(1);

		pty += 110;

		//// add and delete points
		Fl_Button* ap = new Fl_Button(605,pty,80,20,"Export");
		ap->callback((Fl_Callback*)exportSurfelCB,this);
		//Fl_Button* dp = new Fl_Button(690,pty,80,20,"Delete Point");
		//dp->callback((Fl_Callback*)deletePointCB,this);

		pty += 25;
		//// reset the points
		//resetButton = new Fl_Button(735,pty,60,20,"Reset");
		//resetButton->callback((Fl_Callback*)resetCB,this);
		//Fl_Button* loadb = new Fl_Button(605,pty,60,20,"Load");
		//loadb->callback((Fl_Callback*) loadCB, this);
		//Fl_Button* saveb = new Fl_Button(670,pty,60,20,"Save");
		//saveb->callback((Fl_Callback*) saveCB, this);

		//pty += 25;
		//// roll the points
		//Fl_Button* rx = new Fl_Button(605,pty,30,20,"R+X");
		//rx->callback((Fl_Callback*)rpxCB,this);
		//Fl_Button* rxp = new Fl_Button(635,pty,30,20,"R-X");
		//rxp->callback((Fl_Callback*)rmxCB,this);
		//Fl_Button* rz = new Fl_Button(670,pty,30,20,"R+Z");
		//rz->callback((Fl_Callback*)rpzCB,this);
		//Fl_Button* rzp = new Fl_Button(700,pty,30,20,"R-Z");
		//rzp->callback((Fl_Callback*)rmzCB,this);

		//pty+=30;

		//waterBrowser = new Fl_Browser(605, pty, 120, 75, "Wave Type");
		//waterBrowser->type(1);		// select
		//waterBrowser->callback((Fl_Callback*)updateWaterType, this);
		//waterBrowser->add("Sine wave");
		//waterBrowser->add("Heightmap");
		//waterBrowser->add("Simulation");
		//waterBrowser->select(1);

		//pty += 110;

		//amplitude = new Fl_Value_Slider(655, pty, 140, 20, "Amplitude");
		//amplitude->range(0.0, 1.0);
		//amplitude->value(0.1);
		//amplitude->align(FL_ALIGN_LEFT);
		//amplitude->type(FL_HORIZONTAL);

		//pty += 30;

		radius_slider = new Fl_Value_Slider(655, pty, 140, 20, "Radius");
		radius_slider->range(0.01f, 10.0f);
		radius_slider->value(1.0);
		radius_slider->align(FL_ALIGN_LEFT);
		radius_slider->type(FL_HORIZONTAL);

		pty += 30;

		// TODO: add widgets for all of your fancier features here
#ifdef EXAMPLE_SOLUTION
		makeExampleWidgets(this, pty);
#endif

		// we need to make a little phantom widget to have things resize correctly
		Fl_Box* resizebox = new Fl_Box(600, 595, 200, 5);
		widgets->resizable(resizebox);

		widgets->end();
	}
	end();	// done adding to this widget

	// set up callback on idle
	Fl::add_idle((void (*)(void*))idleCB, this);
}

//************************************************************************
//
// * handy utility to make a button into a toggle
//========================================================================
void MyWindow::
togglify(Fl_Button* b, int val)
//========================================================================
{
	b->type(FL_TOGGLE_BUTTON);		// toggle
	b->value(val);		// turned off
	b->selection_color((Fl_Color)3); // yellow when pressed	
	b->callback((Fl_Callback*)damageCB, this);
}

//************************************************************************
//
// *
//========================================================================
void MyWindow::
damageMe()
//========================================================================
{

	myView->damage(1);
}