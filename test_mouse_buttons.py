#!/usr/bin/env python3
"""
Test mouse button support and availability.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mouse_button_support():
    """Test which mouse buttons are available."""
    try:
        from pynput import mouse
        
        logger.info("Testing mouse button availability...")
        
        # Test all mouse buttons
        buttons_to_test = {
            'left': 'mouse.Button.left',
            'right': 'mouse.Button.right', 
            'middle': 'mouse.Button.middle',
            'x1': 'mouse.Button.x1',
            'x2': 'mouse.Button.x2',
        }
        
        available_buttons = {}
        
        for name, button_code in buttons_to_test.items():
            try:
                button = eval(button_code)
                available_buttons[name] = button
                logger.info(f"✅ {name}: {button}")
            except AttributeError as e:
                logger.warning(f"❌ {name}: Not available ({e})")
            except Exception as e:
                logger.error(f"❌ {name}: Error ({e})")
        
        # Test creating a mouse listener
        logger.info("\nTesting mouse listener creation...")
        
        def dummy_callback(x, y, button, pressed):
            logger.info(f"Mouse event: {button} {'pressed' if pressed else 'released'}")
        
        try:
            listener = mouse.Listener(on_click=dummy_callback)
            logger.info("✅ Mouse listener created successfully")
            listener.stop()  # Don't actually start it
        except Exception as e:
            logger.error(f"❌ Failed to create mouse listener: {e}")
        
        return available_buttons
        
    except ImportError as e:
        logger.error(f"Failed to import pynput.mouse: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {}

if __name__ == "__main__":
    print("=== Mouse Button Support Test ===")
    available = test_mouse_button_support()
    
    if available:
        print(f"\nAvailable mouse buttons: {list(available.keys())}")
        
        # Suggest fallback
        if 'x1' not in available:
            if 'middle' in available:
                print("Suggestion: Use 'middle' button as fallback")
            elif 'right' in available:
                print("Suggestion: Use 'right' button as fallback")
    else:
        print("No mouse buttons available or pynput not working")