#include "simulation.h"

#include <iostream>

//// DATA ////

//*

// Parameters for implicit integration
SimulationParameters::SimulationParameters()
: timeStep(0.04), 
//  odeSolver(ODE_EulerExplicit),
  odeSolver(ODE_EulerImplicit),
  rayleighMass(0.01), rayleighStiffness(0.01),
  maxIter(25), tolerance(1e-3),
  youngModulusTop(100000), youngModulusBottom(1000000), poissonRatio(0.4), massDensity(0.01),
  gravity(0,-10,0), pushForce(75, 20, -15), planeRepulsion(10000),
  sphereRepulsion(0),
  fixedHeight(0.05)
{
}

/*/

// Parameters for explicit integration
SimulationParameters::SimulationParameters()
: timeStep(0.0001), 
  odeSolver(ODE_EulerExplicit),
//  odeSolver(ODE_EulerImplicit),
  rayleighMass(0.5), rayleighStiffness(0.01),
  maxIter(25), tolerance(1e-3),
  youngModulusTop(10000), youngModulusBottom(10000), poissonRatio(0.3), massDensity(0.1),
  gravity(0,-10,0), pushForce(75, 20, -15), planeRepulsion(10000),
  sphereRepulsion(0),
  fixedHeight(0.05)
{
}

//*/

SimulationParameters simulation_params;

int verbose = 0;
