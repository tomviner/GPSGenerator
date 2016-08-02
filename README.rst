==========================================================
GPS tracking simulation for a Carrier-Workers coordination
==========================================================

Introduction
-----------------
This a small simulator for generating gps coordinates, simulating a fleet of carriers or workers within a big city.

Objective
-------------
The main objective for this script is to have a flexible, easy to use and customizable stream of gps corrdinates.

Motivation
----------
I need this tool for testing database storing of a big stream of gps tracking.
This is for personal use. I am considering transforming this repo in a small library agnostic to databases, and personal data structures.

Why not TDD
-----------
Ok, i thought that i will use python generator, coroutines, and a scheduler since  the begining and because I really did not  know how deep this rabbithole will be. I decided not to do tdd this time. This script is a test itself. I will rebuild this script with tests, when i really know how to do it.

Performance Issues
------------------
I wanted a small memory footprint script, thats why i decided to do it with generators, at first sight, memory use seems to be ok. but with an intensive use of the script, the cpu is not stable. ( in this toy macbook air )
I will check cpu issues.

Speed Issues
------------
This script generates the stream, I decided to implemente this generator and scheduler trick, to simulate some concurrency, and have fine control of the simulation, in one thread. That' s why this goes slow.


