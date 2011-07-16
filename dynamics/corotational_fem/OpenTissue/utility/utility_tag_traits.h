#ifndef OPENTISSUE_UTILITY_UTILITY_TAG_TRAITS_H
#define OPENTISSUE_UTILITY_UTILITY_TAG_TRAITS_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//

#include <boost/utility.hpp>

namespace OpenTissue
{
  namespace utility
  {

    /**
    Tagging mechanism using SFINAE (substitution-failure-is-not-an-error) principle.

    Example of two classes, one without tags, other with tag member.
    Note that to provide support for the tag, one needs to add the
    typedef *and* the m_tag member variable. Optimized compilers should
    not generate code for if(has_tag(T)) conditionals where T doesn't
    support tags. 

    struct bar {};
    struct foo { typedef void has_tag; int m_tag; foo() : m_tag(23) {} };
    struct blah : public TagSupportedType<int> {};

    int main()
    {
    bar a;
    foo b;

    std::cout << "bar has tag: " << has_tag(a) << std::endl;
    std::cout << "foo has tag: " << has_tag(b) << std::endl;

    if (has_tag(a))
    return 1;

    // always 0
    std::cout << "bar's tag value is " << tag_value(a) << std::endl;

    if (tag_value(b))
    return 1;
    std::cout << "foo's tag value is " << tag_value(b) << std::endl;

    return 0;
    }
    */
    template <typename T> 
    struct TagSupportedType
    {
      typedef void has_tag;   ///< Dummy typedef, required for SFINAE
      typedef T tag_type;     ///< The type of the tag value
      tag_type m_tag;         ///< The tag value
    };
    typedef TagSupportedType<int> default_tag_supported_type;

    template <class T, class Enable = void>
    struct tag_traits { static bool const has_tag = false; typedef int tag_type; };

    template <class T>
    struct tag_traits<T, typename T::has_tag > { static bool const has_tag = true; typedef typename T::tag_type tag_type; };

    template <class T>
    inline bool has_tag(T const & obj, typename boost::disable_if_c< tag_traits<T>::has_tag >::type * dummy = 0) 
    { return false; }
    template <class T>
    inline bool has_tag(T const & obj, typename boost::enable_if_c< tag_traits<T>::has_tag >::type * dummy = 0) 
    { return true; }

    template <typename T>
    inline typename tag_traits<T>::tag_type tag_value(T const & obj, typename boost::disable_if_c< tag_traits<T>::has_tag >::type * dummy = 0) 
    { return 0; }
    template <typename T>
    inline typename tag_traits<T>::tag_type tag_value(T const & obj, typename boost::enable_if_c< tag_traits<T>::has_tag >::type * dummy = 0) 
    { return obj.m_tag; }

    template <class T>
    inline void set_tag(T & /* obj */, typename tag_traits<T>::tag_type const & /* tag */, typename boost::disable_if_c< tag_traits<T>::has_tag >::type * dummy = 0) 
    { }
    template <class T>
    inline void set_tag(T & obj, typename tag_traits<T>::tag_type const & tag, typename boost::enable_if_c< tag_traits<T>::has_tag >::type * dummy = 0) 
    { obj.m_tag = tag; }

  } //namespace utility

} //namespace OpenTissue

// OPENTISSUE_UTILITY_UTILITY_TAG_TRAITS_H
#endif 
