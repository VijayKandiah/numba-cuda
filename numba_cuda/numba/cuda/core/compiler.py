from collections import namedtuple
from numba.core.environment import lookup_environment

CR_FIELDS = ["typing_context",
             "target_context",
             "entry_point",
             "typing_error",
             "type_annotation",
             "signature",
             "objectmode",
             "lifted",
             "fndesc",
             "library",
             "call_helper",
             "environment",
             "metadata",
             # List of functions to call to initialize on unserialization
             # (i.e cache load).
             "reload_init",
             "referenced_envs",
             ]


class CompileResult(namedtuple("_CompileResult", CR_FIELDS)):
    """
    A structure holding results from the compilation of a function.
    """

    __slots__ = ()

    def _reduce(self):
        """
        Reduce a CompileResult to picklable components.
        """
        libdata = self.library.serialize_using_object_code()
        # Make it (un)picklable efficiently
        typeann = str(self.type_annotation)
        fndesc = self.fndesc
        # Those don't need to be pickled and may fail
        fndesc.typemap = fndesc.calltypes = None
        # Include all referenced environments
        referenced_envs = self._find_referenced_environments()
        return (libdata, self.fndesc, self.environment, self.signature,
                self.objectmode, self.lifted, typeann, self.reload_init,
                tuple(referenced_envs))

    def _find_referenced_environments(self):
        """Returns a list of referenced environments
        """
        mod = self.library._final_module
        # Find environments
        referenced_envs = []
        for gv in mod.global_variables:
            gvn = gv.name
            if gvn.startswith("_ZN08NumbaEnv"):
                env = lookup_environment(gvn)
                if env is not None:
                    if env.can_cache():
                        referenced_envs.append(env)
        return referenced_envs

    @classmethod
    def _rebuild(cls, target_context, libdata, fndesc, env,
                 signature, objectmode, lifted, typeann,
                 reload_init, referenced_envs):
        if reload_init:
            # Re-run all
            for fn in reload_init:
                fn()

        library = target_context.codegen().unserialize_library(libdata)
        cfunc = target_context.get_executable(library, fndesc, env)
        cr = cls(target_context=target_context,
                 typing_context=target_context.typing_context,
                 library=library,
                 environment=env,
                 entry_point=cfunc,
                 fndesc=fndesc,
                 type_annotation=typeann,
                 signature=signature,
                 objectmode=objectmode,
                 lifted=lifted,
                 typing_error=None,
                 call_helper=None,
                 metadata=None,  # Do not store, arbitrary & potentially large!
                 reload_init=reload_init,
                 referenced_envs=referenced_envs,
                 )

        # Load Environments
        for env in referenced_envs:
            library.codegen.set_env(env.env_name, env)

        return cr

    @property
    def codegen(self):
        return self.target_context.codegen()

    def dump(self, tab=''):
        print(f'{tab}DUMP {type(self).__name__} {self.entry_point}')
        self.signature.dump(tab=tab + '  ')
        print(f'{tab}END DUMP')
