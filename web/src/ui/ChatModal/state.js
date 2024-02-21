import create from 'zustand';

const useChatModal = create((set) => ({
  path: '',
  visible: false,
  setPath: (_path) => {
                     let value = _path
                  // filter input for only digits and _
                  value = value.replace(/[^1-3_]/g, '')
    set({ path: value })
  },
  setVisible: (_visible) => {
    console.log("VISIBLE", _visible)
    set((state) => ({ visible: _visible }))
  },
}));

export default useChatModal;