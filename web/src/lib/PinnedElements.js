import create from 'zustand'

const useLinkedElementsStore = create((set) => ({
  linkedElements: [],
  addElement: (element) =>
    set((state) => ({ linkedElements: [...state.linkedElements, element] })),
  removeElement: (elementId) =>
    set((state) => ({
      linkedElements: state.linkedElements.filter((el) => el.id !== elementId),
    })),
  clearElements: () => set((state) => ({ linkedElements: [] })),
  linkedElementsHas: (elementId) =>
    set((state) => state.linkedElements.some((el) => el.id === elementId)),
  linkedElementsEmpty: () => set((state) => state.linkedElements.length === 0),
}))

export default useLinkedElementsStore
