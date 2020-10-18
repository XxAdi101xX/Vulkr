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

#include "input_event.h"

namespace vulkr
{

// InputEvent class method implementations
InputEvent::InputEvent( EventSource eventSource) : eventSource{ eventSource }
{}

InputEvent::~InputEvent()
{}

EventSource InputEvent::getEventSource() const
{
	return eventSource;
}

// KeyInputEvent class method implementations
KeyInputEvent::KeyInputEvent(KeyInput input, KeyAction action) : 
	InputEvent{ EventSource::Keyboard },
	input{ input },
	action{ action }
{}

KeyInput KeyInputEvent::getInput() const
{
	return input;
}

KeyAction KeyInputEvent::getAction() const
{
	return action;
}

// MouseInputEvent class method implementations
MouseInputEvent::MouseInputEvent(MouseInput input, MouseAction action, double positionX, double positionY) :
	InputEvent{ EventSource::Mouse },
	input{ input },
	action{ action },
	positionX{ positionX },
	positionY{ positionY }
{}

MouseInput MouseInputEvent::getInput() const
{
	return input;
}

MouseAction MouseInputEvent::getAction() const
{
	return action;
}

double MouseInputEvent::getPositionX() const
{
	return positionX;
}

double MouseInputEvent::getPositionY() const
{
	return positionY;
}

} // namespace vulr