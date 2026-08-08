import pytest
from PySide6.QtCore import Qt, QSize
from vault_gui import VaultWindow

@pytest.fixture
def window(qtbot):
    win = VaultWindow()
    qtbot.addWidget(win)
    win.show()
    return win

def test_vertical_layout_transition(window, qtbot):
    # Set to a narrow width
    window.resize(600, 800)
    qtbot.wait(100)
    
    assert window.split.orientation() == Qt.Vertical
    # Config should be on top (index 0)
    assert window.split.indexOf(window.config_panel) == 0
    assert window.split.indexOf(window.monitor_panel) == 1

def test_scroll_area_activation(window, qtbot):
    # Set a very small height
    window.resize(1000, 400)
    qtbot.wait(100)
    
    # In horizontal mode, if content height > 400, scrollbar should be possible
    # But resizeEvent sets root height based on max(config, monitor) + 150
    # Let's check if the scrollbar exists
    v_bar = window.scroll_area.verticalScrollBar()
    
    # We need to ensure the widget is larger than the viewport
    # Force a small window
    window.resize(1000, 300)
    qtbot.wait(200)
    
    # The minimum height in Horizontal mode is max(h_config, h_monitor) + 150
    # h_config is at least 400 (set in init_ui or sizeHint)
    assert window.scroll_area.widget().minimumHeight() >= 550
    assert v_bar.maximum() > 0  # This means it's scrollable

def test_theme_contrast_switch(window, qtbot):
    # Switch to Light mode
    window.header_mode_combo.setCurrentText("Light")
    qtbot.wait(100)
    
    # Check current theme
    assert "Light" in window.current_theme.name
    
    # Verify some colors (simplified)
    qss = window.styleSheet()
    assert "background-color: #FDF6E3" in qss or "background-color: #F5EFD6" in qss
    assert "color: #073642" in qss or "color: #002B36" in qss
