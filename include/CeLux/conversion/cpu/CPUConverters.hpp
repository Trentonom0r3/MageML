#pragma once
#ifndef CONVERTERS_HPP
#define CONVERTERS_HPP

// Base classes and existing converters
<<<<<<< Updated upstream
#include <cpu/BGRAToRGB.hpp>
#include <cpu/BGRToRGB.hpp>
#include <cpu/CPUConverter.hpp>
#include <cpu/GBRPToRGB.hpp>
#include <cpu/RGBAToRGB.hpp>
#include <cpu/RGBToRGB.hpp>
#include <cpu/YUV420P10ToRGB48.hpp>
#include <cpu/YUV420PToRGB.hpp>
#include <cpu/YUV422P10ToRGB48.hpp>
=======
#include <Celux/conversion/cpu/BGRAToRGB.hpp>
#include <Celux/conversion/cpu/BGRToRGB.hpp>
#include <Celux/conversion/cpu/CPUConverter.hpp>
#include <Celux/conversion/cpu/GBRPToRGB.hpp>
#include <Celux/conversion/cpu/RGBAToRGB.hpp>
#include <Celux/conversion/cpu/RGBToRGB.hpp>
#include <Celux/conversion/cpu/YUV420P10ToRGB48.hpp>
#include <Celux/conversion/cpu/YUV420PToRGB.hpp>
#include <Celux/conversion/cpu/YUV422P10ToRGB48.hpp>
#include <Celux/conversion/cpu/RGB24ToYUV420P.hpp>
#include <Celux/conversion/cpu/AutoToRGB.hpp>
>>>>>>> Stashed changes

// -------------------------------------------------------------------------
// New converters for additional pixel formats
// -------------------------------------------------------------------------

// 8-bit YUV422 -> RGB24
#include <Celux/conversion/cpu/YUV422P8ToRGB24.hpp>

// 12-bit YUV420 -> 48-bit RGB
#include <Celux/conversion/cpu/YUV420P12ToRGB48.hpp>

// 12-bit YUV422 -> 48-bit RGB
#include <Celux/conversion/cpu/YUV422P12ToRGB48.hpp>

// 8-bit YUV444 -> RGB24
#include <Celux/conversion/cpu/YUV444P8ToRGB24.hpp>

// 10-bit YUV444 -> 48-bit RGB
#include <Celux/conversion/cpu/YUV444P10ToRGB48.hpp>

// 12-bit YUV444 -> 48-bit RGB
#include <Celux/conversion/cpu/YUV444P12ToRGB48.hpp>

#endif // Celux/conversion/cpu_CONVERTERS_HPP
