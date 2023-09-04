class TOCSingleton {
  constructor() {
    if (TOCSingleton.instance) {
      return TOCSingleton.instance;
    }

    this.data = {};
    this.failedFetches = new Set(); // To keep track of paths that returned a 404

    TOCSingleton.instance = this;
    return this;
  }

  // Convert different path formats to an array of keys
  _convertPath(path) {
    return path.replace(/_$/, '/_').split('/').filter(Boolean);
  }

  _setNested(obj, pathArr, value) {
    let lastKey = pathArr.pop();
    let deepRef = pathArr.reduce((acc, curr) => acc[curr] = acc[curr] || {}, obj);
    deepRef[lastKey] = value;
  }

  async _fetchData(path) {
    if (this.failedFetches.has(path)) {
      return null; // Do not fetch again if it previously failed
    }

    const res = await fetch(`/api/toc/${path}`);
    if (!res.ok) {
      console.error('Network response was not ok', res);
      if (res.status === 404) {
        this.failedFetches.add(path);
      }
      return null;
    }
    return await res.json();
  }

  getProxy() {
    return new Proxy(this.data, {
      get: (target, prop) => {
        let keys = this._convertPath(prop);

        let ref = target;
        for (let key of keys) {
          if (ref[key] === undefined) {
            this._fetchData(prop).then(newData => {
              if (newData) {
                this._setNested(target, keys, newData);
              }
            });
            return {}; // Return an empty object in the meantime
          }
          ref = ref[key];
        }
        return ref;
      }
    });
  }
}

// Usage:
export const TOC = new TOCSingleton().getProxy();