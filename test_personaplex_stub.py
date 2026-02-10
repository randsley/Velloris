#!/usr/bin/env python3
"""
Stub-only tests for PersonaPlex-7B (no library loading, avoids segfault)
Tests the engine wrapper logic without loading actual PersonaPlex library
"""

import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("PersonaPlex-7B Stub-Mode Tests (No Library Loading)")
print("=" * 70)

def test_voice_data_structure():
    """Test voice data without loading PersonaPlex"""
    from engines.personaplex import PersonaPlexEngine

    print("\n[TEST 1] Voice Data Structure")
    try:
        # Access voices without instantiating (avoids library load)
        voices = PersonaPlexEngine.AVAILABLE_VOICES
        voice_names = list(voices.keys())

        assert len(voice_names) == 18, f"Expected 18 voices, got {len(voice_names)}"

        # Check structure
        for voice_name, voice_file in voices.items():
            assert isinstance(voice_name, str), f"Voice name not string: {voice_name}"
            assert isinstance(voice_file, str), f"Voice file not string: {voice_file}"
            assert voice_file.endswith(".pt"), f"Voice file not .pt: {voice_file}"

        # Count categories
        nat_female = [v for v in voice_names if "natural_female" in v]
        nat_male = [v for v in voice_names if "natural_male" in v]
        var_female = [v for v in voice_names if "varied_female" in v]
        var_male = [v for v in voice_names if "varied_male" in v]

        print(f"  [OK] Voice data structure correct")
        print(f"      - Total voices: {len(voice_names)}")
        print(f"      - Natural female: {len(nat_female)}")
        print(f"      - Natural male: {len(nat_male)}")
        print(f"      - Varied female: {len(var_female)}")
        print(f"      - Varied male: {len(var_male)}")

        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False

def test_class_methods():
    """Test class methods without instantiation"""
    from engines.personaplex import PersonaPlexEngine

    print("\n[TEST 2] Class Methods")
    try:
        # Test get_available_voices class method
        voices = PersonaPlexEngine.get_available_voices()
        assert isinstance(voices, list), "get_available_voices should return list"
        assert len(voices) == 18, f"Expected 18 voices, got {len(voices)}"
        assert all(isinstance(v, str) for v in voices), "All voices should be strings"

        print(f"  [OK] get_available_voices() method works")
        print(f"      - Returns list of {len(voices)} voices")
        print(f"      - Sample voices: {voices[:3]}")

        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_structure():
    """Test configuration setup without PersonaPlex"""
    from config import Config

    print("\n[TEST 3] Configuration Structure")
    try:
        # Check PersonaPlex config
        assert hasattr(Config, 'model'), "Config missing model section"
        assert hasattr(Config.model, 'PERSONAPLEX_DIR'), "Config missing PERSONAPLEX_DIR"

        personaplex_dir = Config.model.PERSONAPLEX_DIR
        print(f"  [OK] PersonaPlex config present")
        print(f"      - Model dir: {personaplex_dir}")

        # Check if models exist
        if personaplex_dir.exists():
            print(f"      - Directory exists: Yes")

            # List model files
            model_files = list(personaplex_dir.glob("*.safetensors"))
            model_files += list(personaplex_dir.glob("*.model"))
            model_files += list(personaplex_dir.glob("*.tgz"))

            print(f"      - Model files found: {len(model_files)}")
        else:
            print(f"      - Directory exists: No (expected if offline)")

        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False

def test_import_structure():
    """Test import structure (without actually importing PersonaPlex library)"""
    print("\n[TEST 4] Import Structure")
    try:
        # Try to import the engine module
        from engines import personaplex
        assert hasattr(personaplex, 'PersonaPlexEngine'), "Module missing PersonaPlexEngine"
        assert hasattr(personaplex, 'HAS_PERSONAPLEX'), "Module missing HAS_PERSONAPLEX"

        print(f"  [OK] Engine module structure correct")
        print(f"      - PersonaPlexEngine class: Present")
        print(f"      - HAS_PERSONAPLEX flag: {personaplex.HAS_PERSONAPLEX}")

        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False

def test_method_signatures():
    """Test method signatures without calling them"""
    from engines.personaplex import PersonaPlexEngine
    import inspect

    print("\n[TEST 5] Method Signatures")
    try:
        # Check public methods exist
        required_methods = [
            '__init__',
            '_load_model',
            '_wrap_text_prompt',
            'generate_s2s_response',
            'unload',
            'get_available_voices',
        ]

        for method_name in required_methods:
            assert hasattr(PersonaPlexEngine, method_name), f"Missing method: {method_name}"
            method = getattr(PersonaPlexEngine, method_name)
            assert callable(method), f"Method not callable: {method_name}"

        print(f"  [OK] All required methods present")
        for method_name in required_methods:
            print(f"      - {method_name}: Present")

        # Check method signatures
        sig = inspect.signature(PersonaPlexEngine.generate_s2s_response)
        params = list(sig.parameters.keys())

        expected_params = ['self', 'audio', 'sr', 'voice_prompt', 'text_prompt', 'streaming']
        assert params == expected_params, f"Signature mismatch: {params} vs {expected_params}"

        print(f"  [OK] Method signatures correct")
        print(f"      - generate_s2s_response has expected parameters")

        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False

# Run all tests
if __name__ == "__main__":
    results = {
        "Voice Data Structure": test_voice_data_structure(),
        "Class Methods": test_class_methods(),
        "Configuration Structure": test_config_structure(),
        "Import Structure": test_import_structure(),
        "Method Signatures": test_method_signatures(),
    }

    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "=" * 70)
        print("SUCCESS: All stub tests passed!")
        print("PersonaPlex-7B implementation structure is correct.")
        print("\nNote: Full inference tests deferred due to PersonaPlex library")
        print("compatibility issues with Windows/Python 3.14.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print(f"FAILED: {total - passed} test(s) failed")
        print("=" * 70)

    sys.exit(0 if passed == total else 1)
