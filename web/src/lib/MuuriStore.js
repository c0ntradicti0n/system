import create from 'zustand'

const useMuuriStore = create((set) => ({
  muuriInstances: [],
  addInstance: (instance) =>
    set((state) => {
      state.muuriInstances.push(instance)
      return state
    }),
  removeInstance: (instance) =>
    set((state) => {
      const index = state.muuriInstances.indexOf(instance)
      if (index > -1) {
        state.muuriInstances.splice(index, 1)
      }
      return state
    }),
  clearInstances: () =>
    set((state) => {
      state.muuriInstances.length = 0
      return state
    }),
}))

export default useMuuriStore
