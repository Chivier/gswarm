#!/usr/bin/env python3
"""
Test script to verify the startup fix for GSwarm model system.
This tests that the model system starts without asyncio errors.
"""

import asyncio
import sys
from pathlib import Path

def test_import():
    """Test that we can import the module without errors"""
    print("🔍 Testing module import...")
    try:
        # This should not raise any asyncio errors
        from gswarm.model.fastapi_head import create_app, state
        print("✅ Module imported successfully")
        return True
    except RuntimeError as e:
        if "no running event loop" in str(e):
            print(f"❌ AsyncIO error during import: {e}")
            return False
        else:
            raise
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_create_app():
    """Test creating the FastAPI app"""
    print("\n🔍 Testing app creation...")
    try:
        from gswarm.model.fastapi_head import create_app
        
        # Test basic app creation
        app = create_app()
        print("✅ Basic app creation successful")
        
        # Test app creation with CLI parameter overrides
        app_with_overrides = create_app(host="0.0.0.0", port=8095, model_port=9010)
        print("✅ App creation with CLI overrides successful")
        
        return True
    except Exception as e:
        print(f"❌ App creation failed: {e}")
        return False


def test_config_override():
    """Test configuration override functionality"""
    print("\n🔍 Testing configuration overrides...")
    try:
        from gswarm.model.fastapi_head import create_app, config
        
        original_host = config.host.host
        original_port = config.host.port
        original_model_port = config.host.model_port
        
        print(f"   Original config: {original_host}:{original_port}, model_port={original_model_port}")
        
        # Create app with overrides
        app = create_app(host="192.168.1.100", port=8095, model_port=9010)
        
        # Check if config was updated
        print(f"   Updated config: {config.host.host}:{config.host.port}, model_port={config.host.model_port}")
        
        if (config.host.host == "192.168.1.100" and 
            config.host.port == 8095 and 
            config.host.model_port == 9010):
            print("✅ Configuration override successful")
            return True
        else:
            print("❌ Configuration override failed")
            return False
            
    except Exception as e:
        print(f"❌ Configuration override test failed: {e}")
        return False


async def test_discovery_startup():
    """Test that discovery works during startup event"""
    print("\n🔍 Testing discovery on startup...")
    try:
        from gswarm.model.fastapi_head import state
        
        # Manually trigger discovery (simulating startup)
        if not state.discovery_completed:
            await state.discover_and_register_models()
            state.discovery_completed = True
            
        print(f"✅ Discovery completed. Found {len(state.models)} models")
        return True
        
    except Exception as e:
        print(f"❌ Discovery test failed: {e}")
        return False


def test_state_initialization():
    """Test that HeadState initializes correctly"""
    print("\n🔍 Testing HeadState initialization...")
    try:
        from gswarm.model.fastapi_head import HeadState
        
        # Create a new state instance (this should not raise asyncio errors)
        test_state = HeadState()
        
        # Check that discovery_completed is properly initialized
        if hasattr(test_state, 'discovery_completed') and test_state.discovery_completed == False:
            print("✅ HeadState initialization successful")
            return True
        else:
            print("❌ HeadState initialization: discovery_completed not properly set")
            return False
            
    except Exception as e:
        print(f"❌ HeadState initialization failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 GSwarm Startup Fix Test Suite")
    print("=" * 50)
    
    tests = [
        test_import,
        test_state_initialization,
        test_create_app,
        test_config_override,
    ]
    
    async_tests = [
        test_discovery_startup,
    ]
    
    passed = 0
    failed = 0
    
    # Run synchronous tests
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    # Run asynchronous tests
    for test in async_tests:
        try:
            if await test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Startup fix is working correctly.")
        print("\n✅ You can now run: gswarm host start --port 8095 --http-port 8096 --model-port 9010")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1) 