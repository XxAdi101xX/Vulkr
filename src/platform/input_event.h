/* Copyright (c) 2020 Adithya Venkatarao
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <GLFW/glfw3.h>
#include "common/vulkan_common.h"

namespace vulkr
{

enum class EventSource
{
	Keyboard,
	Mouse
};

enum class KeyInput
{
	Unknown,
	Space,
	Apostrophe, /* ' */
	Comma,      /* , */
	Minus,      /* - */
	Period,     /* . */
	Slash,      /* / */
	_0,
	_1,
	_2,
	_3,
	_4,
	_5,
	_6,
	_7,
	_8,
	_9,
	Semicolon, /* ; */
	Equal,     /* = */
	A,
	B,
	C,
	D,
	E,
	F,
	G,
	H,
	I,
	J,
	K,
	L,
	M,
	N,
	O,
	P,
	Q,
	R,
	S,
	T,
	U,
	V,
	W,
	X,
	Y,
	Z,
	LeftBracket,  /* [ */
	Backslash,    /* \ */
	RightBracket, /* ] */
	GraveAccent,  /* ` */
	Escape,
	Enter,
	Tab,
	Backspace,
	Insert,
	Delete,
	Right,
	Left,
	Down,
	Up,
	PageUp,
	PageDown,
	Home,
	End,
	CapsLock,
	ScrollLock,
	NumLock,
	PrintScreen,
	Pause,
	F1,
	F2,
	F3,
	F4,
	F5,
	F6,
	F7,
	F8,
	F9,
	F10,
	F11,
	F12,
	KP_0,
	KP_1,
	KP_2,
	KP_3,
	KP_4,
	KP_5,
	KP_6,
	KP_7,
	KP_8,
	KP_9,
	KP_Decimal,
	KP_Divide,
	KP_Multiply,
	KP_Subtract,
	KP_Add,
	KP_Enter,
	KP_Equal,
	LeftShift,
	LeftControl,
	LeftAlt,
	RightShift,
	RightControl,
	RightAlt
};

enum class KeyAction
{
	Press,
	Release,
	Repeat,
	Unknown
};

enum class MouseInput
{
	Left,
	Right,
	Middle,
	Back,
	Forward,
	None
};

enum class MouseAction
{
	Click,
	Release,
	Move,
	Unknown
};

const std::unordered_map<int, KeyInput> keyInputMap =
{
	{GLFW_KEY_SPACE, KeyInput::Space},
	{GLFW_KEY_APOSTROPHE, KeyInput::Apostrophe},
	{GLFW_KEY_COMMA, KeyInput::Comma},
	{GLFW_KEY_MINUS, KeyInput::Minus},
	{GLFW_KEY_PERIOD, KeyInput::Period},
	{GLFW_KEY_SLASH, KeyInput::Slash},
	{GLFW_KEY_0, KeyInput::_0},
	{GLFW_KEY_1, KeyInput::_1},
	{GLFW_KEY_2, KeyInput::_2},
	{GLFW_KEY_3, KeyInput::_3},
	{GLFW_KEY_4, KeyInput::_4},
	{GLFW_KEY_5, KeyInput::_5},
	{GLFW_KEY_6, KeyInput::_6},
	{GLFW_KEY_7, KeyInput::_7},
	{GLFW_KEY_8, KeyInput::_8},
	{GLFW_KEY_9, KeyInput::_9},
	{GLFW_KEY_SEMICOLON, KeyInput::Semicolon},
	{GLFW_KEY_EQUAL, KeyInput::Equal},
	{GLFW_KEY_A, KeyInput::A},
	{GLFW_KEY_B, KeyInput::B},
	{GLFW_KEY_C, KeyInput::C},
	{GLFW_KEY_D, KeyInput::D},
	{GLFW_KEY_E, KeyInput::E},
	{GLFW_KEY_F, KeyInput::F},
	{GLFW_KEY_G, KeyInput::G},
	{GLFW_KEY_H, KeyInput::H},
	{GLFW_KEY_I, KeyInput::I},
	{GLFW_KEY_J, KeyInput::J},
	{GLFW_KEY_K, KeyInput::K},
	{GLFW_KEY_L, KeyInput::L},
	{GLFW_KEY_M, KeyInput::M},
	{GLFW_KEY_N, KeyInput::N},
	{GLFW_KEY_O, KeyInput::O},
	{GLFW_KEY_P, KeyInput::P},
	{GLFW_KEY_Q, KeyInput::Q},
	{GLFW_KEY_R, KeyInput::R},
	{GLFW_KEY_S, KeyInput::S},
	{GLFW_KEY_T, KeyInput::T},
	{GLFW_KEY_U, KeyInput::U},
	{GLFW_KEY_V, KeyInput::V},
	{GLFW_KEY_W, KeyInput::W},
	{GLFW_KEY_X, KeyInput::X},
	{GLFW_KEY_Y, KeyInput::Y},
	{GLFW_KEY_Z, KeyInput::Z},
	{GLFW_KEY_LEFT_BRACKET, KeyInput::LeftBracket},
	{GLFW_KEY_BACKSLASH, KeyInput::Backslash},
	{GLFW_KEY_RIGHT_BRACKET, KeyInput::RightBracket},
	{GLFW_KEY_GRAVE_ACCENT, KeyInput::GraveAccent},
	{GLFW_KEY_ESCAPE, KeyInput::Escape},
	{GLFW_KEY_ENTER, KeyInput::Enter},
	{GLFW_KEY_TAB, KeyInput::Tab},
	{GLFW_KEY_BACKSPACE, KeyInput::Backspace},
	{GLFW_KEY_INSERT, KeyInput::Insert},
	{GLFW_KEY_DELETE, KeyInput::Delete},
	{GLFW_KEY_RIGHT, KeyInput::Right},
	{GLFW_KEY_LEFT, KeyInput::Left},
	{GLFW_KEY_DOWN, KeyInput::Down},
	{GLFW_KEY_UP, KeyInput::Up},
	{GLFW_KEY_PAGE_UP, KeyInput::PageUp},
	{GLFW_KEY_PAGE_DOWN, KeyInput::PageDown},
	{GLFW_KEY_HOME, KeyInput::Home},
	{GLFW_KEY_END, KeyInput::End},
	{GLFW_KEY_CAPS_LOCK, KeyInput::CapsLock},
	{GLFW_KEY_SCROLL_LOCK, KeyInput::ScrollLock},
	{GLFW_KEY_NUM_LOCK, KeyInput::NumLock},
	{GLFW_KEY_PRINT_SCREEN, KeyInput::PrintScreen},
	{GLFW_KEY_PAUSE, KeyInput::Pause},
	{GLFW_KEY_F1, KeyInput::F1},
	{GLFW_KEY_F2, KeyInput::F2},
	{GLFW_KEY_F3, KeyInput::F3},
	{GLFW_KEY_F4, KeyInput::F4},
	{GLFW_KEY_F5, KeyInput::F5},
	{GLFW_KEY_F6, KeyInput::F6},
	{GLFW_KEY_F7, KeyInput::F7},
	{GLFW_KEY_F8, KeyInput::F8},
	{GLFW_KEY_F9, KeyInput::F9},
	{GLFW_KEY_F10, KeyInput::F10},
	{GLFW_KEY_F11, KeyInput::F11},
	{GLFW_KEY_F12, KeyInput::F12},
	{GLFW_KEY_KP_0, KeyInput::KP_0},
	{GLFW_KEY_KP_1, KeyInput::KP_1},
	{GLFW_KEY_KP_2, KeyInput::KP_2},
	{GLFW_KEY_KP_3, KeyInput::KP_3},
	{GLFW_KEY_KP_4, KeyInput::KP_4},
	{GLFW_KEY_KP_5, KeyInput::KP_5},
	{GLFW_KEY_KP_6, KeyInput::KP_6},
	{GLFW_KEY_KP_7, KeyInput::KP_7},
	{GLFW_KEY_KP_8, KeyInput::KP_8},
	{GLFW_KEY_KP_9, KeyInput::KP_9},
	{GLFW_KEY_KP_DECIMAL, KeyInput::KP_Decimal},
	{GLFW_KEY_KP_DIVIDE, KeyInput::KP_Divide},
	{GLFW_KEY_KP_MULTIPLY, KeyInput::KP_Multiply},
	{GLFW_KEY_KP_SUBTRACT, KeyInput::KP_Subtract},
	{GLFW_KEY_KP_ADD, KeyInput::KP_Add},
	{GLFW_KEY_KP_ENTER, KeyInput::KP_Enter},
	{GLFW_KEY_KP_EQUAL, KeyInput::KP_Equal},
	{GLFW_KEY_LEFT_SHIFT, KeyInput::LeftShift},
	{GLFW_KEY_LEFT_CONTROL, KeyInput::LeftControl},
	{GLFW_KEY_LEFT_ALT, KeyInput::LeftAlt},
	{GLFW_KEY_RIGHT_SHIFT, KeyInput::RightShift},
	{GLFW_KEY_RIGHT_CONTROL, KeyInput::RightControl},
	{GLFW_KEY_RIGHT_ALT, KeyInput::RightAlt},
};

const std::unordered_map<int, KeyAction> keyActiontMap =
{
	{GLFW_PRESS, KeyAction::Press},
	{GLFW_RELEASE, KeyAction::Release},
	{GLFW_REPEAT, KeyAction::Repeat}
};

const std::unordered_map<int, MouseInput> mouseInputMap =
{
	{GLFW_MOUSE_BUTTON_1, MouseInput::Left},
	{GLFW_MOUSE_BUTTON_2, MouseInput::Right},
	{GLFW_MOUSE_BUTTON_3, MouseInput::Middle},
	{GLFW_MOUSE_BUTTON_4, MouseInput::Back},
	{GLFW_MOUSE_BUTTON_5, MouseInput::Forward}
};

const std::unordered_map<int, MouseAction> mouseActionMap =
{
	{GLFW_PRESS, MouseAction::Click},
	{GLFW_RELEASE, MouseAction::Release}
};

/* Abstract InputEvent Class */
class InputEvent
{
public:
	InputEvent(EventSource eventSource);

	virtual ~InputEvent() = 0;

	EventSource getEventSource() const;
private:
	EventSource eventSource;
};

class KeyInputEvent : public InputEvent
{
public:
	KeyInputEvent(KeyInput input, KeyAction action);

	~KeyInputEvent() = default;

	KeyInput getInput() const;

	KeyAction getAction() const;
private:
	KeyInput input;

	KeyAction action;
};

class MouseInputEvent : public InputEvent
{
public:
	MouseInputEvent(MouseInput input, MouseAction action, double positionX, double positionY);

	~MouseInputEvent() = default;

	MouseInput getInput() const;

	MouseAction getAction() const;

	double getPositionX() const;

	double getPositionY() const;
private:
	MouseInput input;

	MouseAction action;

	double positionX;

	double positionY;
};

} // namespace vulr